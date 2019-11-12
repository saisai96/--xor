#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace fnnxor
{
	struct FNNXorModel
	{
		XTensor weight1;

		XTensor weight2;

		XTensor b;

		XTensor b1;

		int inp_size;

		int h_size;

		int oup_size;

		int devID;
	};

	struct FNNXorNet
	{
		XTensor hidden_state1;

		XTensor hidden_state2;

		XTensor hidden_state3;

		XTensor output_state1;

		XTensor output_state2;

		XTensor output;
	};

	int FNNXorMain(int argc, const char ** argv);
};
