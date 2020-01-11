#include "FNNXor.h"
#include "../../tensor/function/FHeader.h"
namespace fnnxor
{
	float learningRate = 0.005F;
	int nEpoch = 5000;
	int batch = 64;
	float minmax = 1.0F;

	void Init(FNNXorModel &model);
	void InitGrad(FNNXorModel &model, FNNXorModel &grad);
	void Forward(XTensor &input, FNNXorModel &model, FNNXorNet &net);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void Backward(XTensor &input, XTensor &gold, FNNXorModel &model, FNNXorModel &grad, FNNXorNet &net);
	void Update(XTensor &model, FNNXorModel &grad, float learningRate);
	void CleanGrad(FNNXorModel &grad);

	void Train(float trainDataX[][6], float trainDataY[][3], int dataSizeL, int dataSizeD, FNNXorModel &model);
	void Test(float testDataX[][6], float testDataY[][3], int testDataSizeL, int testDataSizeD, FNNXorModel &model);

	int FNNXorMain(int argc, const char ** argv)
	{
		FNNXorModel model;
		model.inp_size = 6;
		model.h_size = 10;
		model.oup_size = 3;
		const int dataSizeL = 64;
		const int dataSizeD = 6;
		const int testDataSizeL = 64;
		const int testDataSizeD = 6;
		model.devID = 0;
		Init(model);

		float trainDataX[dataSizeL][dataSizeD] = {};
		float trainDataY[dataSizeL][3] = {};
		float testDataX[testDataSizeL][testDataSizeD] = { };
		float testDataY[dataSizeL][3] = {};

		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				int temp1 = i, temp2 = j;
				for (int k = 0; k < 3; k++) {
					trainDataX[i * 8 + j][k] = temp1 % 2;
					trainDataX[i * 8 + j][k + 3] = temp2 % 2;

					testDataX[i * 8 + j][k] = temp1 % 2;
					testDataX[i * 8 + j][k + 3] = temp2 % 2;

					trainDataY[i * 8 + j][k] = (temp1 % 2) ^ (temp2 % 2);

					testDataY[i * 8 + j][k] = (temp1 % 2) ^ (temp2 % 2);
					temp1 = (temp1 - (temp1 % 2)) / 2;
					temp2 = (temp2 - (temp2 % 2)) / 2;
				}
			}
		}

		Train(trainDataX, trainDataY, dataSizeL, dataSizeD, model);

		Test(testDataX, testDataY, testDataSizeL, testDataSizeD, model);
		return 0;
	}//fnnxormain

	void Init(FNNXorModel &model)
	{
		InitTensor2D(&model.weight1, model.inp_size, model.h_size, X_FLOAT, model.devID);
		InitTensor2D(&model.weight2, model.h_size, model.oup_size, X_FLOAT, model.devID);
		InitTensor2D(&model.b, 1, model.h_size, X_FLOAT, model.devID);
		model.weight1.SetDataRand(-minmax, minmax);
		model.weight2.SetDataRand(-minmax, minmax);
		model.b.SetZeroAll();
		printf("Init model finish!\n");
	}//init

	void InitGrad(FNNXorModel &model, FNNXorModel &grad)
	{
		InitTensor(&grad.weight1, &model.weight1);
		InitTensor(&grad.weight2, &model.weight2);
		InitTensor(&grad.b, &model.b);
		InitTensor(&grad.b1, &model.b1);

		grad.inp_size = model.inp_size;
		grad.h_size = model.h_size;
		grad.devID = model.devID;
	}//initgrad

	void Forward(XTensor &input, FNNXorModel &model, FNNXorNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);
		XTensor b = Unsqueeze(model.b, 1, 64);
		//b.Dump(&b, stderr, "b: ");
		b = ReduceSum(b, 0, 1) / 64;
		//b.Dump(&b, stderr, "b: ");
		//net.hidden_state1.Dump(&net.hidden_state1, stderr, "net.hidden_state1: ");
		net.hidden_state2 = net.hidden_state1 + b;
		net.hidden_state3 = HardTanH(net.hidden_state2);
		net.output = MatrixMul(net.hidden_state3, model.weight2);
		//net.output.Dump(&net.output, stderr, "net.output: ");
	}//forward

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
	{
		//output.Dump(&output, stderr, "output: ");
		//gold.Dump(&gold, stderr, "gold: ");
		XTensor tmp = output - gold;
		tmp.Dump(&tmp, stderr, "tmp:");
		//printf("%d\n", output.dimSize[0]);
		_Absolute(&loss, &loss);
		loss = ReduceSum(tmp, 0, 2) / output.dimSize[0];
		loss = Unsqueeze(loss, 0, 1);
		//loss.Dump(&loss, stderr, "loss: ");
		//printf("%d\n", output.dimSize[1]);
		loss = ReduceSum(loss, 1, 2) / output.dimSize[1];
		loss.Dump(&loss, stderr, "loss: ");
	}//mseloss

	void MSELossBackward(XTensor &output, XTensor &gold, XTensor &grad)
	{
		XTensor tmp = output - gold;
		grad = tmp * 2;
		tmp.Dump(&tmp, stderr, "tmpBackward: ");
	}//mselossbackward：因为最后一层只有一个节点，所以loss==(x-y)^2，所以这是特殊的lossbackward

	void Backward(XTensor &input, XTensor &gold, FNNXorModel &model, FNNXorModel &grad, FNNXorNet &net)
	{
		XTensor lossGrad;
		XTensor &dedw1 = grad.weight1;
		XTensor &dedw2 = grad.weight2;
		XTensor &dedb = grad.b;

		XTensor b = Unsqueeze(model.b, 1, 64);
		b = ReduceSum(b, 0, 1) / 64;

		MSELossBackward(net.output, gold, lossGrad);
		dedw2 = MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS);
		XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
		_HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &b);
		//net.hidden_state3.Dump(&net.hidden_state3, stderr, "hidden3: ");
		//net.hidden_state2.Dump(&net.hidden_state2, stderr, "hidden2: ");
		//dedb.Dump(&dedb, stderr, "dedb: ");
		//dedy.Dump(&dedy, stderr, "dedy: ");
		dedw1 = MatrixMul(input, X_TRANS, b, X_NOTRANS);
		//dedw1.Dump(&dedw1, stderr, "dedw1: ");
		b = ReduceSum(b, 0, 1) / 64;
		//b.Dump(&b, stderr, "b: ");
		dedb = Unsqueeze(b, 0, 1);
		//dedb.Dump(&dedb, stderr, "dedb: ");
	}//backward

	void Update(FNNXorModel &model, FNNXorModel &grad, float learningRate)
	{
		//model.weight1.Dump(&model.weight1, stderr, "weight1: ");
		//model.weight2.Dump(&model.weight2, stderr, "weight2: ");
		//grad.weight1.Dump(&grad.weight1, stderr, "grad1: ");
		//grad.weight2.Dump(&grad.weight2, stderr, "grad2: ");
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}//update

	void CleanGrad(FNNXorModel &grad)
	{
		grad.weight1.SetZeroAll();
		grad.weight2.SetZeroAll();
		grad.b.SetZeroAll();
	}//cleangrad

	void Train(float trainDataX[][6], float trainDataY[][3], int dataSizeL, int dataSizeD, FNNXorModel &model)
	{
		printf("prepare data for train\n");

		TensorList inputList;
		TensorList goldList;
		XTensor* inputData = NewTensor2D(dataSizeL, dataSizeD, X_FLOAT, model.devID);
		XTensor* goldData = NewTensor2D(dataSizeL, 3, X_FLOAT, model.devID);
		for (int i = 0; i < dataSizeL; ++i)
		{
			for (int j = 0; j < dataSizeD; ++j)
			{
				inputData->Set2D(trainDataX[i][j], i, j);
			}

			for (int j = 0; j < 3; ++j)
			{
				goldData->Set2D(trainDataY[i][j], i, j);
			}
		}//for
		inputList.Add(inputData);
		goldList.Add(goldData);

		printf("start train\n");
		FNNXorNet net;
		FNNXorModel grad;
		InitGrad(model, grad);
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			//if ((epochIndex + 1) % 50 == 0)
			//	learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				XTensor *input = inputList.GetItem(i);
				XTensor *gold = goldList.GetItem(i);
				//input->Dump(stderr, "input: ");
				//gold->Dump(stderr, "gold: ");
				//printf("begin Forward!\n");
				Forward(*input, model, net);

				XTensor loss;
				//printf("Forward done!\n");
				MSELoss(net.output, *gold, loss);
				//printf("MSELoss done!\n");
				totalLoss = loss.Get1D(0);
				//printf("totalLoss done!\n");
				Backward(*input, *gold, model, grad, net);
				//printf("Backward done!\n");
				Update(model, grad, learningRate);
				//printf("Update done!\n");
				CleanGrad(grad);
				//printf("CleanGrad done!\n");
			}//for
			printf("loss %f\n", totalLoss);
		}//for
	}//train

	void Test(float testDataX[][6], float testDataY[][3], int testDataSizeL, int testDataSizeD, FNNXorModel &model)
	{
		FNNXorNet net;
		XTensor* inputData = NewTensor2D(testDataSizeL, testDataSizeD, X_FLOAT, model.devID);
		float rightnum = 0;
		for (int i = 0; i < testDataSizeL; ++i)
		{
			for (int j = 0; j < testDataSizeD; ++j)
			{
				inputData->Set2D(testDataX[i][j], i, j);
			}
		}
		//inputData->Dump(stderr, "testinput: ");
		Forward(*inputData, model, net);
		int ans = 0;
		for (int i = 0; i < testDataSizeL; ++i)
		{
			float temp = net.output.Get2D(0, 0) + net.output.Get2D(0, 1) * 2 + net.output.Get2D(0, 2) * 4;
			ans = testDataY[i][0] + testDataY[i][1] * 2 + testDataY[i][2] * 4;
			printf("output: %f *** real ans is %d\n", temp, ans);
			if ((ans - temp) < 0.5)
			{
				rightnum++;
			}
		}

		printf("rightpercent is %f, rightnum is %f\n", rightnum / 64, rightnum);
	}//test
}//namespace fnnxor
