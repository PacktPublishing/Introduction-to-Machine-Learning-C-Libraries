#include <stdio.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

// compile with g++ -o lr linear_regression.cpp -std=c++11 -lboost_serialization -lshark -lcblas
#include <iostream>

using namespace std;
using namespace shark;


int main (int argc, char ** argv) {

    Data<RealVector> inputs;
    Data<RealVector> labels;

    importCSV(inputs, "x.csv");
    importCSV(labels, "y.csv");

	RegressionDataset data(inputs, labels);
	
	// trainer and model
    LinearRegression trainer;
    LinearModel<> model;
 
    // train model
    trainer.train(model, data);
  
    // show model parameters
    cout << "intercept: " << model.offset() << endl;
    cout << "matrix: " << model.matrix() << endl;
 
    SquaredLoss<> loss;
    Data<RealVector> prediction = model(data.inputs()); 
    cout << "squared loss: " << loss(data.labels(), prediction) << endl;

	return 0;
}
