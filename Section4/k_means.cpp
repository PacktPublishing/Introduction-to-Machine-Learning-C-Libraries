#include <iostream>
#include <vector>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace std;
using namespace dlib;

int main()
{


    std::vector<matrix<double,2,1>> samples;
    std::vector<matrix<double,2,1>> initial_centers = {{1,1},{2,3}};

    matrix<double,2,1> sample;

    const int num = 10;

    for (int i = 0; i < num; ++i)
    {
        sample(0) = 1.0 + (((double) std::rand() / RAND_MAX) - 0.5);
        sample(1) = 1.0 + (((double) std::rand() / RAND_MAX) - 0.5);
        samples.push_back(sample);
    }

    for (int i = 0; i < num; ++i)
    {
        sample(0) = 2 + (((double) std::rand() / RAND_MAX) - 0.5);
        sample(1) = 3 + (((double) std::rand() / RAND_MAX) - 0.5);
        samples.push_back(sample);
    }


    pick_initial_centers(2, initial_centers, samples);

    find_clusters_using_kmeans(samples,initial_centers, 10);

    for (const auto & it : initial_centers) {
        std::cout << "cluster x: " << it(0) << " , y: " << it(1) << std::endl;
    }


}



