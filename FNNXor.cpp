#include "FNNXor.h"
#include "../../tensor/function/FHeader.h"
namespace fnnxor
{
	float learningRate = 0.3F;
	int nEpoch = 100;
	float minmax = 0.01F;

	void Init(FNNXorModel &model);
	void InitGrad(FNNXorModel &model, FNNXorModel &grad);
	void Forward(XTensor &input, FNNXorModel &model, FNNXorNet &net);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void Backward(XTensor &input, XTensor &gold, FNNXorModel &model, FNNXorModel &grad, FNNXorNet &net);
	void Update(XTensor &model, FNNXorModel &grad, float learningRate);
	void CleanGrad(FNNXorModel &grad);

	void Train(float trainDataX[][2], float trainDataY[][8], int dataSizeL, int dataSizeD, FNNXorModel &model);
	void Test(float testDataX[][2], float testDataY[][8], int testDataSizeL, int testDataSizeD, FNNXorModel &model);

	int FNNXorMain(int argc, const char ** argv) 
	{
		FNNXorModel model;
		model.inp_size = 2;
		model.h_size = 20;
		model.oup_size = 8;
		const int dataSizeL = 64;
		const int dataSizeD = 2;
		const int testDataSizeL = 64;
		const int testDataSizeD = 2;
		model.devID = 0;
		Init(model);

		float trainDataX[dataSizeL][dataSizeD] = {};
		float trainDataY[dataSizeL][8] = {};
		float testDataX[testDataSizeL][testDataSizeD] = { };
		float testDataY[dataSizeL][8] = {};

		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				trainDataX[i * 8 + j][0] = i;
				trainDataX[i * 8 + j][1] = j;
				trainDataY[i * 8 + j][i ^ j] = 1;

				testDataX[i * 8 + j][0] = i;
				testDataX[i * 8 + j][1] = j;
				testDataY[i * 8 + j][i ^ j] = 1;
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

		grad.inp_size = model.inp_size;
		grad.h_size = model.h_size;
		grad.devID = model.devID;
	}//initgrad

	void Forward(XTensor &input, FNNXorModel &model, FNNXorNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);
		net.hidden_state2 = net.hidden_state1 + model.b;
		net.hidden_state3 = HardTanH(net.hidden_state2);
		net.output = MatrixMul(net.hidden_state3, model.weight2);
	}//forward

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
	{
		//output.Dump(&output, stderr, "output: ");
		//gold.Dump(&gold, stderr, "gold: ");
		XTensor tmp = output - gold;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
		//loss.Dump(&loss, stderr, "loss: ");
	}//mseloss

	void MSELossBackward(XTensor &output, XTensor &gold, XTensor &grad)
	{
		XTensor tmp = output - gold;
		grad = tmp * 2;
		//tmp.Dump(&tmp, stderr, "tmp: ");
	}//mselossbackward：因为最后一层只有一个节点，所以loss==(x-y)^2，所以这是特殊的lossbackward

	void Backward(XTensor &input, XTensor &gold, FNNXorModel &model, FNNXorModel &grad, FNNXorNet &net)
	{
		XTensor lossGrad;
		XTensor &dedw1 = grad.weight1;
		XTensor &dedw2 = grad.weight2;
		XTensor &dedb = grad.b;

		MSELossBackward(net.output, gold, lossGrad);
		MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);
		XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
		_HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);
		net.hidden_state3.Dump(&net.hidden_state3, stderr, "hidden3: ");
		net.hidden_state2.Dump(&net.hidden_state2, stderr, "hidden2: ");
		dedb.Dump(&dedb, stderr, "dedb: ");
		dedy.Dump(&dedy, stderr, "dedy: ");
		dedw1 = MatrixMul(input, X_TRANS, dedb, X_NOTRANS);
		dedw1.Dump(&dedw1, stderr, "dedw1: ");
	}//backward

	void Update(FNNXorModel &model, FNNXorModel &grad, float learningRate)
	{
		model.weight1.Dump(&model.weight1, stderr, "weight1: ");
		model.weight2.Dump(&model.weight2, stderr, "weight2: ");
		grad.weight1.Dump(&grad.weight1, stderr, "grad1: ");
		grad.weight2.Dump(&grad.weight2, stderr, "grad2: ");
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

	void Train(float trainDataX[][2], float trainDataY[][8], int dataSizeL, int dataSizeD, FNNXorModel &model)
	{
		printf("prepare data for train\n");

		TensorList inputList;
		TensorList goldList;
		for (int i = 0; i < dataSizeL; ++i)
		{
			XTensor* inputData = NewTensor2D(1, dataSizeD, X_FLOAT, model.devID);
			for (int j = 0; j < dataSizeD; ++j)
			{
				inputData->Set2D(trainDataX[i][j] / 100, 0, j);
			}
			inputList.Add(inputData);

			XTensor* goldData = NewTensor2D(1, 8, X_FLOAT, model.devID);
			for (int j = 0; j < 8; ++j)
			{
				//if (trainDataY[i][j] != 1)
				//{
				//	goldData->Set2D(trainDataY[i] / 100, 0, 0);
				//}
				goldData->Set2D(trainDataY[i][j] / 60, 0, j);
			}
			goldList.Add(goldData);
		}//for

		printf("start train\n");
		FNNXorNet net;
		FNNXorModel grad;
		InitGrad(model, grad);
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				XTensor *input = inputList.GetItem(i);
				XTensor *gold = goldList.GetItem(i);
				//input->Dump(stderr, "input: ");
				//gold->Dump(stderr, "gold: ");
				Forward(*input, model, net);

				XTensor loss;
				MSELoss(net.output, *gold, loss);

				totalLoss += loss.Get1D(0);

				Backward(*input, *gold, model, grad, net);
				
				Update(model, grad, learningRate);

				CleanGrad(grad);
			}//for
			printf("loss %f\n", totalLoss / inputList.count);
		}//for
	}//train

	void Test(float testDataX[][2], float testDataY[][8], int testDataSizeL, int testDataSizeD, FNNXorModel &model)
	{
		FNNXorNet net;
		XTensor* inputData = NewTensor2D(1, testDataSizeD, X_FLOAT, model.devID);
		float rightnum = 0;
		for (int i = 0; i < testDataSizeL; ++i)
		{
			for (int j = 0; j < testDataSizeD; ++j)
			{
				inputData->Set2D(testDataX[i][j] / 100, 0, j);
			}
			//inputData->Dump(stderr, "testinput: ");
			Forward(*inputData, model, net);
			int ans = 0;
			float temp = net.output.Get2D(0, 0) * 60;
			for (int j = 0; j < 8; j++)
			{
				//printf("output: %f\n", net.output.Get2D(0, j) * 60);
				if (temp < (net.output.Get2D(0, j) * 60))
				{
					temp = net.output.Get2D(0, j) * 60;
					ans = j;
				}
			}
			if (testDataY[i][ans] == 1)
			{
				rightnum++;
			}
			//float ans = net.output.Get2D(0, 0)  * 60;
			/*printf("ans is : %d\n", ans);*/
		}
		printf("rightpercent is %f, rightnum is %f\n", rightnum / 64, rightnum);
	}//test
}//namespace fnnxor