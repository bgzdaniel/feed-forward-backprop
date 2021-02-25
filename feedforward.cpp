#include <iostream>
#include <fstream>
#include <cmath>

#include <armadillo>

using namespace arma;
using namespace std;

// converts from high endian to low endian
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void show_distribution(colvec &data)
{
    double mean = arma::mean(data);
    double stddev = arma::stddev(data);
    uvec histogram = hist(data, 50);
    std::vector<std::string> symbols = {"_", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"};
    int divider = std::max((int)std::ceil((float)max(histogram) / 8), 1);
    cout << "mean: " << mean << "\tstddev: " << stddev << "\t[";
    for (auto it = histogram.begin(); it != histogram.end(); ++it)
    {
        int index = (*it) / divider;
        cout << symbols[index];
    }
    cout << "]" << endl
         << endl;
}

double sig(double t)
{
    return (1 / (1 + exp(-t)));
}

int main()
{
    arma_rng::set_seed(10);
    // get the data of the image file from the mnist dataset
    mat images;
    ifstream imageFile("mnistData/train-images-idx3-ubyte", std::ios::binary);
    if (imageFile.is_open())
    {
        int magicNumber = 0;
        imageFile.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        int numberOfImages = 0;
        imageFile.read((char *)&numberOfImages, sizeof(numberOfImages));
        numberOfImages = reverseInt(numberOfImages);
        int rows = 0;
        imageFile.read((char *)&rows, sizeof(rows));
        rows = reverseInt(rows);
        int cols = 0;
        imageFile.read((char *)&cols, sizeof(cols));
        cols = reverseInt(cols);
        images.zeros(numberOfImages, (rows * cols));
        for (int i = 0; i < numberOfImages; ++i)
        {
            for (int j = 0; j < (rows * cols); ++j)
            {
                unsigned char pixel = 0;
                imageFile.read((char *)&pixel, sizeof(pixel));
                images(i, j) = (double)pixel;
            }
        }
        imageFile.close();
    }

    // whitening of the image data
    // for each pixel -> pixel - mean
    // for each pixel -> pixel - stddeviation

    colvec means = mean(images, 1);
    images -= repmat(means, 1, 784);

    colvec stddevs = stddev(images, 0, 1);
    images /= repmat(stddevs, 1, 784);

    // get the data of the label file from the mnist dataset
    rowvec labels;
    ifstream labelFile("mnistData/train-labels-idx1-ubyte", std::ios::binary);
    if (labelFile.is_open())
    {
        int magicNumber = 0;
        labelFile.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        int numberOfItems = 0;
        labelFile.read((char *)&numberOfItems, sizeof(numberOfItems));
        numberOfItems = reverseInt(numberOfItems);
        labels.zeros(numberOfItems);
        for (int i = 0; i < numberOfItems; ++i)
        {
            unsigned char label = 0;
            labelFile.read((char *)&label, sizeof(label));
            labels(i) = (double)label;
        }
        labelFile.close();
    }

    //switch images
    images = images.t();

    // set weights for each layer
    int neuronAmount = 2;
    mat weightsL0to1(neuronAmount, images.n_rows, fill::randu);
    weightsL0to1 /= 100;
    mat weightsL1to2(1, neuronAmount, fill::randu);
    weightsL1to2 /= 100;

    // set batches
    int batchSize = 2;
    mat miniBatch = images.cols(0, batchSize - 1);
    mat batchLabels = labels.subvec(0, batchSize - 1);

    // feed-forward
    mat activationL1 = weightsL0to1 * miniBatch;
    mat activationL1Gradient = activationL1;
    activationL1.transform([](double val) { return sig(val); });
    activationL1Gradient.transform([](double val) { return sig(val) * (1 - sig(val)); });
    mat activationL2 = weightsL1to2 * activationL1;
    mat activationL2Gradient = activationL2;
    activationL2.transform([](double val) { return sig(val); });
    activationL2Gradient.transform([](double val) { return sig(val) * (1 - sig(val)); });
    mat cost = (activationL2 - batchLabels) * 2;

    miniBatch.brief_print("miniBatch:");
    weightsL0to1.brief_print("weights layer 0 to 1:");
    activationL1.brief_print("activationL1:");
    weightsL1to2.brief_print("weights layer 1 to 2:");
    activationL2.brief_print("activationL2:");
    batchLabels.brief_print("batchLabels:");
    cost.brief_print("cost:");

    // get error

}