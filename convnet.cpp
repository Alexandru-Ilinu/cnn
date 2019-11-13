#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <ctime>

using namespace std; 

const float learningRate = 1.0f; 
const int NR_EXAMPLES_READ = 100; 
const int IMAGE_HEIGHT = 28; 
const int IMAGE_WIDTH = 28; 
const bool GRAYSCALE = true; 
const int NR_CHANNELS = GRAYSCALE ? 1 : 3; 
const int NR_EPOCHS = 100; 
const float RATIO_TRAIN_TEST = 0.9; 

enum Mode { TRAIN_AND_TEST, TRAIN, TRAIN_CONTINUE, TEST };

struct Neuron
{
	float a; 
	float delta; 
	
	Neuron() { a = 0; delta = 0; }
}; 

template<class T>
T ***init3DMatrix(int H, int W, int C)
{
	int i, j; 
	T ***m = new T **[H]; 
	for (i = 0; i < H; i++)
	{
		m[i] = new T *[W]; 
		for (j = 0; j < W; j++)
		{
			m[i][j] = new T[C]; 
		}
	}
	return m; 
}

template<class T>
void delete3DMatrix(T ***m, int H, int W, int C)
{
	int i, j; 
	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			delete [] m[i][j]; 
		}
		delete [] m[i]; 
	}
	delete [] m; 
}

float genRandomWeight()
{
	return (rand() / (float) RAND_MAX - 0.5); 
}

float sigmoidFunction(float x)
{
	return 1/(float)(1 + pow(M_E, -x)); 
}

float sigmoidFunctionDerivative(float x)
{
	float s = sigmoidFunction(x); 
	return s * (1 - s); 
}

class Data
{

}; 

class Data3D : public Data
{
	public:
		int H, W, C; 
		Neuron ***neurons; 
	
	Data3D(int h, int w, int c) : 
		H(h), W(w), C(c)
	{
		neurons = init3DMatrix<Neuron>(H, W, C); 
	}
	
	~Data3D()
	{
		delete3DMatrix<Neuron>(neurons, H, W, C); 
	}
}; 

class Data1D : public Data
{
	public:
		int nrNeurons; 
		Neuron *neurons; 
	
	Data1D(int nr) : nrNeurons(nr)
	{
		neurons = new Neuron[nr]; 
	}
	
	~Data1D()
	{
		delete [] neurons; 
	}
}; 

class Filter
{
	public:
		int H, W, C; 
		float ***weights; 
		float bias; 
	
	Filter(int h, int w, int c)
		: H(h), W(w), C(c)
	{
		weights = new float **[2 * H + 1]; 
		for (int i = 0; i < 2 * H + 1; i++)
		{
			weights[i] = new float *[2 * W + 1]; 
			for (int j = 0; j < 2 * W + 1; j++)
			{
				weights[i][j] = new float[C]; 
				for (int k = 0; k < C; k++)
				{
					weights[i][j][k] = genRandomWeight(); 
				}
			}
		}
		bias = genRandomWeight(); 
	}
	
	~Filter()
	{
		for (int i = 0; i < 2 * H + 1; i++)
		{
			for (int j = 0; j < 2 * W + 1; j++)
			{
				delete [] weights[i][j]; 
			}
			delete [] weights[i]; 
		}
		delete [] weights; 
	}
	
}; 

class Layer
{
	public:
		virtual void forward() = 0; 
		virtual void backProp() = 0; 
		virtual void updateWeights() = 0; 
}; 

class ConvolutionalLayer : public Layer
{
	public:
		Data3D *inputData; 
		Data3D *outputData; 
		int nrFilters; 
		int filterH, filterW; 
		Filter **filters; 
		
	ConvolutionalLayer(Data3D *in, Data3D **out, int nrF, int h, int w)
		: inputData(in), nrFilters(nrF), filterH(h), filterW(w)
	{
		int filterC = inputData->C; 
		int outH = inputData->H; 
		int outW = inputData->W; 
		*out = new Data3D(outH, outW, nrFilters); 
		outputData = *out; 
		filters = new Filter *[nrFilters]; 
		for (int i = 0; i < nrFilters; i++)
		{
			filters[i] = new Filter(filterH, filterW, filterC); 
		}
	}
	
	~ConvolutionalLayer()
	{
		for (int i = 0; i < nrFilters; i++)
		{
			delete filters[i]; 
		}
		delete [] filters; 
		delete outputData; 
	}
	
	inline float getActivationPadding(int i, int j, int k)
	{
		int H = inputData->H; 
		int W = inputData->W; 
		
		if (i < 0 || i >= H || j < 0 || j >= W) 
		{
			return 0; 
		}
		else
		{
			return inputData->neurons[i][j][k].a; 
		}
	}
	
	inline float getDeltaPadding(int i, int j, int k)
	{
		int H = outputData->H; 
		int W = outputData->W; 
		
		if (i < 0 || i >= H || j < 0 || j >= W)
		{
			return 0; 
		}
		else
		{
			return outputData->neurons[i][j][k].delta; 
		}
	}
	
	void forward()
	{
		float s; 
		float activation; 
		int outH = outputData->H; 
		int outW = outputData->W; 
		int filterC = inputData->C; 
		
		for (int l = 0; l < nrFilters; l++)
		{		
			for (int i = 0; i < outH; i++)
			{
				for (int j = 0; j < outW; j++)
				{
					s = filters[l]->bias; 
					for (int ii = 0; ii < 2 * filterH + 1; ii++)
					{
						for (int jj = 0; jj < 2 * filterW + 1; jj++)
						{
							for (int kk = 0; kk < filterC; kk++)
							{
								activation = getActivationPadding(i + ii - filterH, j + jj - filterW, kk); 
								s += filters[l]->weights[ii][jj][kk] * activation; 
							}
						}
					}
					
					outputData->neurons[i][j][l].a = s; 
				}
			}
		}
	}
	
	void backProp()
	{
		float s; 
		float delta; 
		float w; 
		
		int H = inputData->H; 
		int W = inputData->W; 
		int filterC = inputData->C; 
		
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				for (int k = 0; k < filterC; k++)
				{
					s = 0; 
					
					for (int ii = -filterH; ii <= filterH; ii++)
					{
						for (int jj = -filterW; jj <= filterW; jj++)
						{
							for (int kk = 0; kk < nrFilters; kk++)
							{
								delta = getDeltaPadding(i + ii, j + jj, kk); 
								w = filters[kk]->weights[filterH - ii][filterW - jj][k]; 
								s += delta * w; 
							}
						}
					}
					
					inputData->neurons[i][j][k].delta = s; 
				}
			}
		}
	}
	
	void updateWeights()
	{
		float s; 
		float a; 
		float d; 
		int filterC = inputData->C; 
		int H = outputData->H; 
		int W = outputData->H; 
		
		for (int l = 0; l < nrFilters; l++)
		{
			for (int ii = 0; ii < 2 * filterH + 1; ii++)
			{
				for (int jj = 0; jj < 2 * filterW + 1; jj++)
				{
					for (int kk = 0; kk < filterC; kk++)
					{
						// update weight 
						// filters[l].weights[ii][jj][kk]
						s = 0; 
						for (int i = 0; i < H; i++)
						{
							for (int j = 0; j < W; j++)
							{
								a = getActivationPadding(i + ii - filterH, j + jj - filterW, kk); 
								d = outputData->neurons[i][j][l].delta; 
								s += a * d; 
							}
						}
						filters[l]->weights[ii][jj][kk] -= learningRate * s; 
					}
				}
			}
			
			// update bias for filter l, filters[l].bias
			s = 0; 
			for (int i = 0; i < H; i++)
			{
				for (int j = 0; j < W; j++)
				{
					s += outputData->neurons[i][j][l].delta; 
				}
			}
			filters[l]->bias -= learningRate * s; 
		}
	}
}; 

class ReLULayer : public Layer
{
	public:
		Data3D *inputData; 
		Data3D *outputData; 
		int H, W, C; 
	
	ReLULayer(Data3D *in, Data3D **out) 
		: inputData(in)
	{
		H = inputData->H; 
		W = inputData->W; 
		C = inputData->C; 
		*out = new Data3D(H, W, C); 
		outputData = *out; 
	}
	
	~ReLULayer()
	{
		delete outputData; 
	}
	
	void forward()
	{
		float x; 
		
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				for (int k = 0; k < C; k++)
				{
					x = inputData->neurons[i][j][k].a; 
					outputData->neurons[i][j][k].a = sigmoidFunction(x); 
				}
			}
		}
	}
	
	void backProp()
	{
		float x; 
		float deriv; 
		float dlt; 
		
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				for (int k = 0; k < C; k++)
				{
					x = inputData->neurons[i][j][k].a; 
					dlt = outputData->neurons[i][j][k].delta; 
					deriv = sigmoidFunctionDerivative(x); 
					inputData->neurons[i][j][k].delta = deriv * dlt; 
				}
			}
		}
	}
	
	void updateWeights() {}
}; 

class PoolingLayer : public Layer
{
	public:
		Data3D *inputData; 
		Data3D *outputData; 
		int size; 
		int outH, outW, outC; 
		
	PoolingLayer(Data3D *in, Data3D **out, int sz = 2)
		: inputData(in), size(sz)
	{
		outH = inputData->H / size; 
		outW = inputData->W / size; 
		outC = inputData->C; 
		*out = new Data3D(outH, outW, outC); 
		outputData = *out; 
	}
	
	~PoolingLayer()
	{
		delete outputData; 
	}
	
	void forward()
	{
		float s; 
		
		for (int i = 0; i < outH; i++)
		{
			for (int j = 0; j < outW; j++)
			{
				for (int k = 0; k < outC; k++)
				{
					s = 0; 
					
					for (int ii = 0; ii < size; ii++)
					{
						for (int jj = 0; jj < size; jj++)
						{
							s += inputData->neurons[i * size + ii][j * size + jj][k].a; 
						}
					}
					s = s / (float) (size * size); 
					
					outputData->neurons[i][j][k].a = s; 
				}
			}
		}
	}
	
	void backProp()
	{
		float fact = 1 / (float) (size * size); 
		
		for (int k = 0; k < outC; k++)
		{
			for (int i = 0; i < outH; i++)
			{
				for (int j = 0; j < outW; j++)
				{
					for (int ii = 0; ii < size; ii++)
					{
						for (int jj = 0; jj < size; jj++)
						{
							inputData->neurons[i * size + ii][j * size + jj][k].delta = fact * outputData->neurons[i][j][k].delta; 
						}
					}
					
				}
			}
		}
	}
	
	void updateWeights() {}
}; 

class LinearisationLayer : public Layer
{
	public:
		Data3D *inputData; 
		Data1D *outputData; 
		int nrNeurons; 
		
	LinearisationLayer(Data3D *in, Data1D **out)
		: inputData(in)
	{
		nrNeurons = inputData->H * inputData->W * inputData->C; 
		*out = new Data1D(nrNeurons); 
		outputData = *out; 
	}
	
	~LinearisationLayer()
	{
		delete outputData; 
	}
	
	void forward()
	{
		int idx; 
		int H = inputData->H; 
		int W = inputData->W; 
		int C = inputData->C; 
		
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				for (int k = 0; k < C; k++)
				{
					idx = i * W * C + j * C + k; 
					outputData->neurons[idx].a = inputData->neurons[i][j][k].a; 
				}
			}
		}
	}
	
	void backProp()
	{
		int idx = 0; 
		int H = inputData->H; 
		int W = inputData->W; 
		int C = inputData->C; 
		
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				for (int k = 0; k < C; k++)
				{
					idx = i * W * C + j * C + k; 
					inputData->neurons[i][j][k].delta = outputData->neurons[idx].delta; 
				}
			}
		}
	}
	
	void updateWeights() {}
}; 

class FullyConnectedLayer : public Layer
{
	public:
		Data1D *inputData; 
		Data1D *outputData; 
		
		float **weights; 
		float *biases; 
		
		int nrNeurIn, nrNeurOut; 
		
		FullyConnectedLayer(
				Data1D *in, 
				Data1D *out
			) : inputData(in), 
				outputData(out)
		{
			nrNeurIn = inputData->nrNeurons; 
			nrNeurOut = outputData->nrNeurons; 
			
			weights = new float *[nrNeurOut]; 
			for (int i = 0; i < nrNeurOut; i++)
			{
				weights[i] = new float[nrNeurIn]; 
				for (int j = 0; j < nrNeurIn; j++)
				{
					weights[i][j] = genRandomWeight(); 
				}
			}
			biases = new float[nrNeurOut]; 
			for (int i = 0; i < nrNeurOut; i++)
			{
				biases[i] = genRandomWeight(); 
			}
		}
		
		~FullyConnectedLayer()
		{
			delete [] biases; 
			for (int i = 0; i < nrNeurOut; i++)
			{
				delete [] weights[i]; 
			}
			delete [] weights; 
		}
		
		void forward()
		{
			float s; 
			
			for (int i = 0; i < nrNeurOut; i++)
			{
				s = biases[i]; 
				for (int j = 0; j < nrNeurIn; j++)
				{
					s += weights[i][j] * inputData->neurons[j].a; 
				}
				outputData->neurons[i].a = s; 
			}
		}
		
		void backProp()
		{
			float s; 
			
			for (int j = 0; j < nrNeurIn; j++)
			{
				s = 0; 
				
				for (int i = 0; i < nrNeurOut; i++)
				{
					s += weights[i][j] * outputData->neurons[i].delta; 
				}
				
				inputData->neurons[j].delta = s; 
			}
		}
		
		void updateWeights()
		{
			float aj, di; 
			
			for (int i = 0; i < nrNeurOut; i++)
			{
				di = outputData->neurons[i].delta; 
				
				for (int j = 0; j < nrNeurIn; j++)
				{
					aj = inputData->neurons[j].a; 
					weights[i][j] -= learningRate * aj * di; 
				}
				biases[i] -= learningRate * di; 
			}
		}
}; 

class ReLULayer1D : public Layer
{
	public:
		Data1D *inputData; 
		Data1D *outputData; 
		int nrNeurons; 
		
	ReLULayer1D(Data1D *in, Data1D **out)
		: inputData(in)
	{
		nrNeurons = inputData->nrNeurons; 
		*out = new Data1D(nrNeurons); 
		outputData = *out; 
	}
	
	~ReLULayer1D()
	{
		delete outputData; 
	}
	
	void forward()
	{
		for (int i = 0; i < nrNeurons; i++)
		{
			outputData->neurons[i].a = sigmoidFunction(inputData->neurons[i].a); 
		}
	}
	
	void backProp()
	{
		float ak; 
		float fderiv; 
		
		for (int i = 0; i < nrNeurons; i++)
		{
			ak = inputData->neurons[i].a; 
			fderiv = sigmoidFunctionDerivative(ak); 
			inputData->neurons[i].delta = outputData->neurons[i].delta * fderiv; 
		}
	}
	
	void updateWeights() {}
}; 


/////////////////////////////////////////////////
class Example
{
	public:
		int H, W, C; 
		float ***pixels; 
		int labelId; 
	
	Example(int h, int w, int c)
		: H(h), W(w), C(c)
	{
		pixels = init3DMatrix<float>(H, W, C); 
	}
	
	~Example()
	{
		delete3DMatrix<float>(pixels, H, W, C); 
	}
}; 

class DataSet
{
	public:
		int nrExamples; 
		Example **examples; 
	
	DataSet()
	{
		nrExamples = NR_EXAMPLES_READ; 
		examples = new Example *[nrExamples]; 
		for (int i = 0; i < nrExamples; i++)
		{
			examples[i] = new Example(28, 28, 1); 
		}
		readData(); 
	}
	
	~DataSet()
	{
		for (int i = 0; i < nrExamples; i++)
		{
			delete examples[i]; 
		}
		delete [] examples; 
	}
	
	void readData()
	{
		ifstream images; 
		images.open("train-images.idx3-ubyte", ios::binary); 
		ifstream labels; 
		labels.open("train-labels.idx1-ubyte", ios::binary); 
		
		unsigned char a; 
		for (int i = 0; i < 16; i++)
		{
			images.read((char *) &a, 1); 
		}
		
		for (int i = 0; i < 8; i++)
		{
			labels.read((char *) &a, 1); 
		}
		
		for (int k = 0; k < nrExamples; k++)
		{
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					images.read((char *) &a, 1); 
					examples[k]->pixels[i][j][0] = a / (float) 255; 
				}
			}
		}
		
		for (int k = 0; k < nrExamples; k++)
		{
			labels.read((char *) &a, 1); 
			examples[k]->labelId = (int) a; 
		}
		
		images.close(); 
		labels.close(); 
	}
}; 

vector<int> initIdxVect(int length)
{
	vector<int> randIdx; 
	for (int i = 0; i < length; i++)
	{
		randIdx.push_back(i); 
	}
	return randIdx; 
}

void labelToVector(int label, float *a, int nrLabels)
{
	for (int i = 0; i < nrLabels; i++)
	{
		a[i] = 0; 
	}
	a[label] = 1; 
}

int vectorToLabel(float *v, int nrLabels)
{
	float max = v[0], imax = 0; 
	for (int i = 1; i < nrLabels; i++)
	{
		if (v[i] > max)
		{
			max = v[i]; 
			imax = i; 
		}
	}
	return imax; 
}


class ConvolutionalNeuralNetwork
{
	public:
		static const int nrLayers = 12; 
		int inputH, inputW, inputC; 
		int nrClasses; 
		Layer *layers[nrLayers]; 
		Data *data[nrLayers]; 
		static const int inputIdx = 0; 
		static const int outputIdx = nrLayers - 1; 
		
		ConvolutionalNeuralNetwork(
			int nrCl, 
			int h, 
			int w, 
			int c
		)
		{
			nrClasses = nrCl; 
			inputH = h; 
			inputW = w; 
			inputC = c; 
			
			// input data (28 x 28 x 1)
			data[0] = new Data3D(inputH, inputW, inputC); 
			// first layer - a convolutional layer - 3 3x3x1 filters
			layers[1] = new ConvolutionalLayer((Data3D *) data[0], (Data3D **) &(data[1]), 3, 1, 1); 
			// data out size: 28 x 28 x 3
			
			// pooling layer - half
			layers[2] = new PoolingLayer((Data3D *) data[1], (Data3D **) &data[2], 2); 
			// data out size: 14 x 14 x 3
			
			layers[3] = new ReLULayer((Data3D *) data[2], (Data3D **) &data[3]); 
			
			layers[4] = new ConvolutionalLayer((Data3D *) data[3], (Data3D **)&data[4], 7, 2, 2); 
			
			layers[5] = new PoolingLayer((Data3D *)data[4], (Data3D **)&data[5], 2); 
			
			layers[6] = new ReLULayer((Data3D *)data[5], (Data3D **)&data[6]); 
			
			layers[7] = new LinearisationLayer((Data3D *)data[6], (Data1D **)&data[7]); 
			
			data[8] = new Data1D(100); 
			layers[8] = new FullyConnectedLayer((Data1D *) data[7], (Data1D *)data[8]); 
			
			layers[9] = new ReLULayer1D((Data1D *)data[8], (Data1D **) &data[9]); 
			
			data[10] = new Data1D(nrClasses); 
			layers[10] = new FullyConnectedLayer((Data1D *)data[9], (Data1D *)data[10]); 
			
			layers[11] = new ReLULayer1D((Data1D *)data[10], (Data1D **) &data[11]); 

		}
		
		~ConvolutionalNeuralNetwork()
		{
			delete layers[11]; 
			delete layers[10]; 
			delete data[10]; 
			delete layers[9]; 
			delete layers[8]; 
			delete data[8]; 
			delete layers[7]; 
			delete layers[6]; 
			delete layers[5]; 
			delete layers[4]; 
			delete layers[3]; 
			delete layers[2]; 
			delete layers[1]; 
			delete data[0]; 
		}
		
		void applyInput(float ***in)
		{
			int H = inputH; 
			int W = inputW; 
			int C = inputC; 
			Data3D *inputInNetwork = (Data3D *) data[0]; 
			
			for (int i = 0; i < H; i++)
			{
				for (int j = 0; j < W; j++)
				{
					for (int k = 0; k < C; k++)
					{
						inputInNetwork->neurons[i][j][k].a = in[i][j][k]; 
					}
				}
			}
		}
		
		void getOutput(float *networkOutput)
		{
			for (int i = 0; i < nrClasses; i++)
			{
				networkOutput[i] = ((Data1D *) data[11])->neurons[i].a; 
			}
		}
		
		void forward(float ***in)
		{
			applyInput(in); 
			for (int i = 1; i < nrLayers; i++)
			{
				layers[i]->forward(); 
			}
		}
		
		void backProp(float *target)
		{
			// compute deltas for the last layer
			{
				Data1D *outputOfNetwork = (Data1D *) data[outputIdx]; 
				
				for (int i = 0; i < nrClasses; i++)
				{
					outputOfNetwork->neurons[i].delta = outputOfNetwork->neurons[i].a - target[i]; 
				}
			}
			
			for (int i = nrLayers - 1; i >= 1; i--)
			{
				layers[i]->backProp(); 
			}
		}
		
		void backProp(int targetLabelId)
		{
			float *a = new float[nrClasses]; 
			labelToVector(targetLabelId, a, nrClasses); 
			backProp(a); 
			delete [] a; 
		}
		
		void updateWeights()
		{
			for (int i = 1; i < nrLayers; i++)
			{
				layers[i]->updateWeights(); 
			}
		}	
		
		void train(DataSet *dataset, int nrEpochs, float ratioTrainTest)
		{
			int nrTrainingExamples = (int) (dataset->nrExamples * ratioTrainTest); 
			vector<int> randIndices = initIdxVect(nrTrainingExamples); 
			float *targetOutput = new float[nrClasses]; 
			int ridx; 

			for (int epoch = 0; epoch < nrEpochs; epoch++)
			{
				cout << "epoch: " << epoch << endl; 
				
				// randomise input examples
				random_shuffle(randIndices.begin(), randIndices.end()); 
				
				for (int trEx = 0; trEx < nrTrainingExamples; trEx++)
				{					
					ridx = randIndices[trEx]; 
					forward(dataset->examples[ridx]->pixels); 
					labelToVector(dataset->examples[ridx]->labelId, targetOutput, nrClasses); 		
					backProp(targetOutput); 
					updateWeights(); 
				}
			}
			
			delete [] targetOutput; 
		}
		
		void test(DataSet *dataset, float ratioTrainTest)
		{
			int nrTrainingExamples = (int) (dataset->nrExamples * ratioTrainTest); 
			int outputLabel; 
			float *networkOutput = new float[nrClasses]; 
			
			cout << "Training data: --------------------------------------------" << endl; 
			for (int i = 0; i < dataset->nrExamples; i++)
			{
				if (i == nrTrainingExamples)
				{
					cout << "Test data: ----------------------------" << endl; 
				}

				forward(dataset->examples[i]->pixels); 
				getOutput(networkOutput); 
				outputLabel = vectorToLabel(networkOutput, nrClasses); 
				
				cout << "Example nr. " << i << endl; 
				cout << "True label: " << dataset->examples[i]->labelId << endl; 
				cout << "Network's output" << endl << "for digit: \t"; 
				for (int j = 0; j < nrClasses; j++)
				{
					printf("%d\t", j); 
				}
				cout << endl; 
				
				cout << "\t\t"; 
				for (int j = 0; j < nrClasses; j++)
				{
					printf("%.3f \t", networkOutput[j]); 
				}
				cout << endl; 
				
				if (outputLabel == dataset->examples[i]->labelId)
				{
					cout << "GUESSED" << endl; 
				}
				else
				{
					cout << "NOT GUESSED" << endl; 
				}
				cout << endl; 

			}
			
			delete [] networkOutput; 
		}
		
		void writeConvolutionalLayerToFile(int layerId, ofstream &outfile)
		{
			ConvolutionalLayer *cl = (ConvolutionalLayer *) layers[layerId]; 
			for (int filterId = 0; filterId < cl->nrFilters; filterId++)
			{
				Filter *filter = cl->filters[filterId]; 
				outfile << filter->bias << endl; 
				for (int i = 0; i < 2 * filter->H + 1; i++)
				{
					for (int j = 0; j < 2 * filter->W + 1; j++)
					{
						for (int k = 0; k < filter->C; k++)
						{
							outfile << filter->weights[i][j][k] << " "; 
						}
						outfile << "  \t  "; 
					}
					outfile << endl; 
				}
				
				outfile << endl; 
			}
		}
		
		void writeFullyConnectedLayerToFile(int layerId, ofstream &outfile)
		{
			FullyConnectedLayer *layer = (FullyConnectedLayer *) layers[layerId]; 
			for (int i = 0; i < layer->nrNeurOut; i++)
			{
				outfile << layer->biases[i] << endl; 
				for (int j = 0; j < layer->nrNeurIn; j++)
				{
					outfile << layer->weights[i][j] << " "; 
				}
				outfile << endl; 
			}
		}
		
		void writeToFile(const char *filename)
		{
			ofstream outfile; 
			outfile.open(filename, ios_base::out); 
			
			writeConvolutionalLayerToFile(1, outfile); 
			writeConvolutionalLayerToFile(4, outfile); 
			writeFullyConnectedLayerToFile(8, outfile); 
			writeFullyConnectedLayerToFile(10, outfile); 
			
			outfile.close(); 
		}
		
		void readConvolutionalLayerFromFile(int layerId, ifstream &infile)
		{
			ConvolutionalLayer *cl = (ConvolutionalLayer *) layers[layerId]; 
			
			for (int filterId = 0; filterId < cl->nrFilters; filterId++)
			{
				Filter *filter = cl->filters[filterId]; 
				infile >> filter->bias; 
				for (int i = 0; i < 2 * filter->H + 1; i++)
				{
					for (int j = 0; j < 2 * filter->W + 1; j++)
					{
						for (int k = 0; k < filter->C; k++)
						{
							infile >> filter->weights[i][j][k]; 
						}
					}
				}
			}
		}
		
		void readFullyConnectedLayerFromFile(int layerId, ifstream &infile)
		{
			FullyConnectedLayer *layer = (FullyConnectedLayer *) layers[layerId]; 
			
			for (int i = 0; i < layer->nrNeurOut; i++)
			{
				infile >> layer->biases[i]; 
				for (int j = 0; j < layer->nrNeurIn; j++)
				{
					infile >> layer->weights[i][j]; 
				}
			}
		}
		
		void readFromFile(const char *filename)
		{
			ifstream infile; 
			infile.open(filename, ios_base::in); 
			
			readConvolutionalLayerFromFile(1, infile); 
			readConvolutionalLayerFromFile(4, infile); 
			readFullyConnectedLayerFromFile(8, infile); 
			readFullyConnectedLayerFromFile(10, infile); 
			
			infile.close(); 
		}
}; 


int main()
{
	srand(time(NULL)); 
	
	int H = 28, W = 28; // image sizes
	int C = 1; 			// nr channels
	
	int nrClasses = 10; 
	int nrEpochs = NR_EPOCHS; 
	float ratioTrainTest = RATIO_TRAIN_TEST; 
	Mode mode = TRAIN_AND_TEST; 
	char *filename = "dbg1.txt"; 
	char *filename2 = "merge.txt"; 
	
	ConvolutionalNeuralNetwork cnn(nrClasses, H, W, C); 
	DataSet dataset; 
	
	if (mode == TRAIN_AND_TEST)
	{
		cnn.train(&dataset, nrEpochs, ratioTrainTest); 
		cnn.test(&dataset, ratioTrainTest); 
		cnn.writeToFile(filename2); 
	}
	else if (mode == TRAIN)
	{
		cnn.train(&dataset, nrEpochs, ratioTrainTest); 
		cnn.writeToFile(filename2); 
	}
	else if (mode == TRAIN_CONTINUE)
	{
		cnn.readFromFile(filename); 
		cnn.train(&dataset, nrEpochs, ratioTrainTest); 
		cnn.writeToFile(filename2); 
	}
	else if (mode == TEST)
	{
		cnn.readFromFile(filename); 
		cnn.test(&dataset, ratioTrainTest); 
	}
	
	return 0; 
}

// Alexandru Ilinu
