#include <fstream>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>

//g++ k_means.cpp -o kmeans_test -O3 -std=c++11 -larmadillo -lmlpack -lboost_serialization && ./kmeans_test 

int main() {

    int k = 2; 
    int dim = 2; 
    int samples = 50;
    int max_iter = 10;
     
    arma::mat data(dim, samples, arma::fill::zeros);

    // create data
    int i = 0;
    for(; i < samples / 2; ++i)
    {
        data.col(i) = arma::vec({1, 1}) + 0.25*arma::randn<arma::vec>(dim);
    }
    for(; i < samples; ++i)
    {
        data.col(i) = arma::vec({2, 3}) + 0.25*arma::randn<arma::vec>(dim);
    }


    //cluster the data
    arma::Row<size_t> clusters;
    arma::mat centroids;

    mlpack::kmeans::KMeans<> mlpack_kmeans(max_iter);

    mlpack_kmeans.Cluster(data, k, clusters, centroids);


    centroids.print("Centroids:");


    return 0;
}