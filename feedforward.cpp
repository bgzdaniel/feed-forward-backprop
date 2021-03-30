#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include <armadillo>

#define ARMA_NO_DEBUG

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
    arma_rng::set_seed(7);
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

    // convert labels to One Hot enconding
    mat oneHot(10, labels.n_elem, fill::zeros);
    for (int i = 0; i < labels.n_elem; ++i)
    {
        int index = labels(i);
        oneHot(index, i) = 1;
    }

    // switch images matrix
    images = images.t();

    // set weights for each layer
    int neuronAmountL1 = 100;
    mat weightsL0to1(neuronAmountL1, images.n_rows, fill::randu);
    weightsL0to1 /= 100;
    int neuronAmountL2 = 10; // must be 10 to work
    mat weightsL1to2(neuronAmountL2, neuronAmountL1, fill::randu);
    weightsL1to2 /= 100;
    double learnRate = 0.02;

    // set batches
    int batchSize = 100;

    int epochAmount = 5;

    colvec errorsEpoch(epochAmount, fill::zeros);

    std::uniform_int_distribution<int> unif(0, images.n_cols - 1);
    std::default_random_engine re;

    for (int epoch = 0; epoch < epochAmount; epoch++)
    {
        double error = 0;
        int correct = 0;
        int wrong = 0;
        colvec errors(images.n_cols, fill::zeros);

        for (int j = 0; j < 60000; j++)
        {
            int colOne = unif(re);
            int colTwo = unif(re);
            images.swap_cols(colOne, colTwo);
            oneHot.swap_cols(colOne, colTwo);
        }
        for (int i = 0; i < images.n_cols; i += batchSize)
        {
            mat miniBatch = images.cols(i, i + batchSize - 1);
            mat batchLabels = oneHot.cols(i, i + batchSize - 1);

            // feed-forward
            mat activationL1 = weightsL0to1 * miniBatch;
            mat activationL1Gradient = activationL1;
            activationL1.transform([](double val) { return val /*sig(val)*/; });
            activationL1Gradient.transform([](double val) { return 1 /*sig(val) * (1 - sig(val))*/; });
            mat activationL2 = weightsL1to2 * activationL1;
            mat activationL2Gradient = activationL2;
            activationL2.transform([](double val) { return val /*sig(val)*/; });
            activationL2Gradient.transform([](double val) { return 1 /*sig(val) * (1 - sig(val))*/; });

            //backpropagation

            //determine error % activationGradient of each layer
            mat errorL2 = ((activationL2 - batchLabels) * 2) % activationL2Gradient;
            mat errorL1 = (weightsL1to2.t() * errorL2) % activationL1Gradient;

            // update weights
            mat L1to2_change = learnRate * ((errorL2 * activationL1.t()) / batchSize);
            weightsL1to2 -= L1to2_change;
            mat L0to1_change = learnRate * ((errorL1 * miniBatch.t()) / batchSize);
            weightsL0to1 -= L0to1_change;

            // L1to2_change.brief_print("L1to2_change:");
            // // L0to1_change.brief_print("L0to1_change:");
            // activationL2.col(i).brief_print("activationL2:");
            // batchLabels.col(i).brief_print("batchLabels:");
            // // activationL1.col(i).brief_print("activationL1:");

            // check results and print to console
            error = mean(mean(activationL2 - batchLabels, 0));
            if (error >= 1)
            {
                cout << "error is equal/over 1!!!" << endl;
                return 0;
            }
            errors(i) = error;

            for (int i = 0; i < batchSize; i++)
            {
                vec a = activationL2.col(i);
                vec b = batchLabels.col(i);
                if (index_max(a) == index_max(b))
                {
                    ++correct;
                }
                else
                {
                    ++wrong;
                }
            }
            cout << "epoch: " << epoch + 1 << " | ";
            cout << "correct: " << correct << " | ";
            cout << "wrong: " << wrong << " | ";
            std::cout << std::fixed;
            std::cout << std::setprecision(4);
            cout << "success rate: " << ((double)correct / (correct + wrong)) * 100 << "% | ";
            std::cout << std::setprecision(10);
            cout << "error: " << error << endl;
            std::cout << std::defaultfloat;
        }

        errorsEpoch(epoch) = mean(errors);
    }

    errorsEpoch.print("errors of each epoch:");
}