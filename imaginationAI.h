#pragma once
#include <vector>

namespace imaginationAI {
	class imaginationNetwork;
	class Matrix;
	class output;

	class imaginationNetwork {
	private:
		std::vector<int> layerBP;
		int layer, inputAmount, outputAmount;
		double lr;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		double(*activationFunction)(double); // f(x) = 1 / (1 + exp(-x))
		double(*derivativeActivationFunction)(double);
	public:
		imaginationNetwork(int layers[], int layerAmount, double(*activationFunction)(double),
			double(*derActFunc)(double), double learningRate);
		output result(double input_array[]);
		void train(double input_array[], double target_array[]);
		void save(std::string filePath);
		static imaginationNetwork load(std::string filePath, double(*actFunc)(double),
		double(*derActFunc)(double));
		void setLearnRate(double learnRate);
	private:
		void setNewWeightMatrix(int index, Matrix m);
		void setNewBiasMatrix(int index, Matrix m);
	};

	class Matrix {
	private:
		int m_rows, m_cols;
		std::vector<std::vector<double>> m_matrix;
	public:
		Matrix(int rows, int columns);
		void randomize(int min, int max);

		void addNum(double value);
		void subtractNum(double value);
		void divideNum(double value);
		void multiplyNum(double value);

		void addMatrix(Matrix m);
		void subtractMatrix(Matrix m);
		void divideMatrix(Matrix m);
		void multiplyMatrix(Matrix m);

		static Matrix matrixMultiply(Matrix m1, Matrix m2);
		static Matrix matrixSubtraction(Matrix m1, Matrix m2);
		static Matrix map(Matrix m, double(*func)(double));
		static Matrix fromArray(double inpArr[], int length);
		static Matrix matrixProduct(Matrix m1, Matrix m2);
		static Matrix transpose(Matrix m);
		static output getOutput(Matrix m);

		double getMatrixValue(int i1, int i2);
		int getRows();
		int getCols();

		void setValue(int i1, int i2, double value);
	private:
		double getRandomDouble(int min, int max);
	};

	class output {
	private:
		std::vector<double> outputs;
	public:
		void push_back(double value);
		double getValue(int index);
	};
}