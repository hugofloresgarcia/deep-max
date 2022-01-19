/// @file
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@copyright	Copyright 2021 Hugo Flores García. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "../shared/signal_routing_objects.h"

#include <torch/script.h>
#include <torch/torch.h>

using namespace c74::min;
using DeepModel = torch::jit::script::Module;

class deeplearning : public object<deeplearning>, public vector_operator<>{
public:
	MIN_DESCRIPTION {"Deep learning in max!"};
	MIN_TAGS {"AI, machine learning, neural networks, deep learning"};
	MIN_AUTHOR {"Hugo Flores García"};

	// inlets, outlets
	inlet<> m_in1 {this, "(signal) input", "signal"};
	outlet<> m_out1 {this, "(signal) output", "signal"};

	// how many samples are we going to collect?
	size_t m_frame_size = 1024;
	fifo<double> m_in_fifo {m_frame_size};
	fifo<double> m_out_fifo {m_frame_size};

	// allocate input and output tensors
	torch::Tensor m_input_tensor {
		torch::zeros(m_frame_size, torch::kDouble)
	};
	torch::Tensor m_output_tensor {
		torch::zeros(m_frame_size, torch::kDouble)
	};

	queue<> m_deferrer{ this, 
		MIN_FUNCTION {
			// drain audio here
			do_forward();
			return {};
		}
	};

	timer<> m_deliverer { this, 
		MIN_FUNCTION {
			m_deferrer.set();
			return {};
		}
	};

	DeepModel m_model;

	void do_forward() {
		// collect samples from the queue
		for (size_t i = 0; i < m_frame_size; i++) {
			// throw if dequeing fails (something went wrong)
			if (!m_in_fifo.try_dequeue(*m_input_tensor[i].data_ptr<double>()))
				throw std::runtime_error("Failed to dequeue from the input FIFO");
		}

		m_output_tensor = m_model.forward({m_input_tensor}).toTensor();

		// push output samples to the output queue
		for (size_t i = 0; i < m_frame_size; i++) {
			m_out_fifo.enqueue(*m_output_tensor[i].data_ptr<double>());
		}
	}

	// collect samples from the operator() thread into the deep learning thread
	void in_collect_samples(audio_bundle& bundle, size_t channel) {
		double* in = bundle.samples(channel);
		for (size_t i = 0; i < bundle.frame_count(); ++i) {
			// if queue is full, it's time to deliver the samples to the model
			if (!m_in_fifo.try_enqueue(in[i]))
				m_deliverer.delay(0);
		}
	}

	// collect samples from the deep learning thread into the operator() thread
	void out_collect_samples(audio_bundle& bundle, size_t channel){
		double* out = bundle.samples(channel);
		for (size_t i = 0; i < bundle.frame_count(); ++i) {
			if (!m_out_fifo.try_dequeue(out[i]))
		 		out[i] = 0.0; // dropped sample :(
		}
	}
	
	// loads the PyTorch model into memory
	void load_model(const std::string& path) {
		m_model = DeepModel(torch::jit::load(path, torch::kCPU));
		m_model.eval();
	}

	void operator()(audio_bundle in, audio_bundle out) {
		in_collect_samples(in, 0);
		out_collect_samples(out, 0);
	}

};

MIN_EXTERNAL(deeplearning);
