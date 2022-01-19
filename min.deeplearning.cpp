/// @file
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@copyright	Copyright 2021 Hugo Flores García. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "../shared/signal_routing_objects.h"

// #include "onnxruntime_c_api.h"

#include <torch/script.h>
#include <torch/torch.h>

// Here we are using the "c74::min" namespace in a header file.
// This is not a generally advisable practice in C++ but in limited cases such as this it makes sense.
// This header file will only be included in Min projects and the code below would be
// very onerous if nearly every symbol we used required a fully-qualified name.

using namespace c74::min;
using DeepModel = torch::jit::script::Module;

// The deeplearning~ object inherits all of it's attributes and messages from the signal_routing_base class

class deeplearning : public object<deeplearning>, public vector_operator<>{
public:
	MIN_DESCRIPTION {"Deep learning in max!"};
	MIN_TAGS {"AI, machine learning, neural networks, deep learning"};
	MIN_AUTHOR {"Hugo Flores García"};
	// MIN_RELATED {"panner~, matrix~"};

	inlet<> m_in1 {this, "(signal) input", "signal"};
	outlet<> m_out1 {this, "(signal) output", "signal"};

	// how many samples are we going to collect?
	size_t m_frame_size = 1024;
	fifo<double> m_in_fifo;
	fifo<double> m_out_fifo;

	DeepModel m_model;

	// collect samples from the operator() thread into the deep learning thread
	void in_collect_samples(const audio_bundle& bundle, const size_t channel) {
		double* in = bundle.samples(channel);
		for (size_t i = 0; i < bundle.frame_count(); ++i) {

			// if the queue is full, 
			if (!m_fifo.try_enqueue(in[i]))
			{
				
			}
		}
	}

	// collect samples from the deep learning thread into the operator() thread
	void out_collect_samples(const audio_bundle &){}
	
	// loads the PyTorch model into memory
	void load_model(const std::string& path) {
		m_model = DeepModel(torch::jit::load(path, torch::kCPU));
		m_model.eval();
	}

	void operator()(audio_bundle in, audio_bundle out) {
		double* in_samples = in.samples(0);
		double* out_samples = out.samples(0);

		torch::Tensor in_tensor = torch::from_blob(in.samples(), 
						 { in.channel_count(), in.frame_count() },
						 torch::TensorOptions().dtype(torch::kDouble));
		
		torch::Tensor out_tensor = m_model.forward({tensor}).toTensor();

		timer<> deliverer { this, 
			MIN_FUNCTION {
				deferrer.set();
				return {};
			}
		};

		queue<> deferrer{ this, 
			MIN_FUNCTION {
				// drain audio here
			}
		};


		
	}
};

MIN_EXTERNAL(deeplearning);
