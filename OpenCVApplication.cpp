// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <numeric> 
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

long int random(void);

typedef struct {
    char* name;
    int label;
} Train_Element;

#define MAX_LINE_LENGTH 1024

void testOpenImage()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src;
        src = imread(fname);
        imshow("opened image", src);
        waitKey();
    }
}

void testOpenImagesFld()
{
    char folderName[MAX_PATH];
    if (openFolderDlg(folderName) == 0)
        return;
    char fname[MAX_PATH];
    FileGetter fg(folderName, "bmp");
    while (fg.getNextAbsFile(fname))
    {
        Mat src;
        src = imread(fname);
        imshow(fg.getFoundFileName(), src);
        if (waitKey() == 27) //ESC pressed
            break;
    }
}

void testColor2Gray()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

        int height = src.rows;
        int width = src.cols;

        Mat_<uchar> dst(height, width);

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                Vec3b v3 = src(i, j);
                uchar b = v3[0];
                uchar g = v3[1];
                uchar r = v3[2];
                dst(i, j) = (r + g + b) / 3;
            }
        }

        imshow("original image", src);
        imshow("gray image", dst);
        waitKey();
    }
}

//Functie care citeste csv-ul orifinal cu test
char** read_test(const char* path, int* numRows) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return NULL;
    }

    char** list = NULL;
    int capacity = 10;
    *numRows = 0;

    list = (char**)malloc(capacity * sizeof(char*));
    if (!list) {
        fclose(file);
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    char line[MAX_LINE_LENGTH];
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';
        //memorie pentru randul nou
        list[*numRows] = _strdup(line);
        (*numRows)++;

        //resize daca e necesar
        if (*numRows >= capacity) {
            capacity *= 2;
            char** newList = (char**)realloc(list, capacity * sizeof(char*));
            if (!newList) {
                fclose(file);
                fprintf(stderr, "Memory reallocation failed.\n");
                return NULL;
            }
            list = newList;
        }
    }

    fclose(file);
    return list;
}

//Functie care citeste csv-ul orifinal cu train
Train_Element* read_train(const char* path, int* numRows) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return NULL;
    }

    Train_Element* list = NULL;
    int capacity = 10;
    *numRows = 0;

    list = (Train_Element*)malloc(capacity * sizeof(Train_Element));
    if (!list) {
        fclose(file);
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {

        line[strcspn(line, "\n")] = '\0';


        char* token = strtok(line, ",");
        if (!token) continue;


        list[*numRows].name = _strdup(token);


        token = strtok(NULL, ",");
        if (!token) {
            free(list[*numRows].name);
            continue;
        }
        list[*numRows].label = atoi(token);
        //schimbam etichetele de la 0-5 la 1-6
        list[*numRows].label++;

        (*numRows)++;

        if (*numRows >= capacity) {
            capacity *= 2;
            Train_Element* newList = (Train_Element*)realloc(list, capacity * sizeof(Train_Element));
            if (!newList) {
                fclose(file);
                fprintf(stderr, "Memory reallocation failed.\n");
                return NULL;
            }
            list = newList;
        }
    }

    fclose(file);
    return list;
}

void freeTest(char** list, int size) {
    if (!list) return;

    for (int i = 0; i < size; i++) {
        free(list[i]);
    }
    free(list);
}

void freeTrain(Train_Element* list, int size) {
    if (!list) return;
    for (int i = 0; i < size; i++) {
        free(list[i].name);
    }
    free(list);
}

//Functie preluata din laborator
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

    //computes histogram maximum
    int max_hist = 0;
    for (int i = 0; i < hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height / max_hist;
    int baseline = hist_height - 1;

    // calculate the width of each bin
    int bin_width = imgHist.cols / hist_cols;

    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x * bin_width, baseline);
        Point p2 = Point(x * bin_width, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
    }

    imshow(name, imgHist);
    waitKey(0);
}

// Input: o lista de struct cu nume poza si eticheta -> Output: o lista cu etichete generate random
int* generare_etichete(int size_list) {
    srand(time(NULL));

    //facem un nou vector pentru noile etichete generate random
    int* etichete_generate = NULL;
    int size_etichete_generate = size_list;
    etichete_generate = (int*)malloc(size_etichete_generate * sizeof(int));

    for (int i = 0; i < size_etichete_generate; i++) {
        //generare eticheta random 1-6
        int random_number = 1 + rand() % (6 - 1 + 1);
        etichete_generate[i] = random_number;
    }
    return etichete_generate;
}

//calcul acuratete
float calcul_acuratete(Train_Element* original, int* generate, int size) {
    int ok = 0;
    for (int i = 0; i < size; i++) {
        if (original[i].label == generate[i]) {
            ok++;
        }
    }
    float acc = (float)ok / size;
    return acc;
}

void afisare_acuratete(Train_Element* test_list, int* etichete_generate, int size) {
    //calculam acuratetea in functie de cate etichete generate in mod random corect avem (pentru test_list)
    float acc = calcul_acuratete(test_list, etichete_generate, size);
    printf("Acuratete: %f \n", acc);
}


void show_split_train(Train_Element* new_train_list, Train_Element* new_test_list, int new_train_size, int new_test_size) {
    for (int i = 0; i < new_train_size; i++) {
        printf("Train list / Nume: %s, Eticheta: %d\n", new_train_list[i].name, new_train_list[i].label);
    }
    for (int i = 0; i < new_test_size; i++) {
        printf("Test list / Nume: %s, Eticheta: %d\n", new_test_list[i].name, new_test_list[i].label);

    }
    printf("Train list size: %d \n", new_train_size);
    printf("Test list size: %d \n", new_test_size);

}

void histograme_test_train(Train_Element* train_list, Train_Element* test_list, int train_size, int test_size) {
    //Nou vector care contine doar etichetele din train
    int* etichete_train = NULL;
    etichete_train = (int*)malloc(train_size * sizeof(int));
    for (int i = 0; i < train_size; i++) {
        etichete_train[i] = train_list[i].label;
    }

    //vector care contine frecventa etichetelor din train
    int size = 6;
    int* frecv_etichete = NULL;
    frecv_etichete = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        frecv_etichete[i] = std::count(etichete_train, etichete_train + train_size, i + 1);
    }
    for (int i = 0; i < size; i++) {
        printf("%d ", frecv_etichete[i]);
    }
    //desenam histograma in functie de frecventa etichetelor
    showHistogram("Train Histogram", frecv_etichete, 200, 250);

    printf("\n");

    //Nou vector care contine doar etichetele din test
    int* etichete_test = NULL;
    etichete_test = (int*)malloc(test_size * sizeof(int));
    for (int i = 0; i < test_size; i++) {
        etichete_test[i] = test_list[i].label;
    }

    int* frecv_etichete_test = NULL;
    frecv_etichete_test = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        frecv_etichete_test[i] = std::count(etichete_test, etichete_test + test_size, i + 1);
    }
    for (int i = 0; i < size; i++) {
        printf("%d ", frecv_etichete_test[i]);
    }

    showHistogram("Test Histogram", frecv_etichete_test, 200, 250);

    free(etichete_train);
    free(etichete_test);
}

void afisare_etichete(Train_Element* test_list, int size) {
    int* etichete_generate = generare_etichete(size);
    for (int i = 0; i < size; i++) {
        printf("Poza: %s cu eticheta generata %d \n", test_list[i].name, etichete_generate[i]);
    }
}


void clearInputBuffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}


char* get_path() {
    printf("Enter the path to the original train.csv file: ");

    char* path = (char*)malloc(1024 * sizeof(char));
    if (!path) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    if (!fgets(path, 1024, stdin)) {
        free(path);
        perror("Error reading input");
        exit(EXIT_FAILURE);
    }

    // Removing trailing newline character, if any
    size_t len = strlen(path);
    if (len > 0 && path[len - 1] == '\n') {
        path[len - 1] = '\0';
    }

    return path;
}

void showConfusionMatrix(Train_Element* test_list, int* generated_labels, int size, int num_classes) {
    int** matrix = (int**)calloc(num_classes, sizeof(int*));
    int* totals = (int*)calloc(num_classes, sizeof(int)); // For storing total counts per actual class

    for (int i = 0; i < num_classes; i++) {
        matrix[i] = (int*)calloc(num_classes, sizeof(int));
    }

    // Fill the confusion matrix and count totals
    for (int i = 0; i < size; i++) {
        int actual = test_list[i].label - 1; // Adjust for 0-based index
        int predicted = generated_labels[i] - 1;
        matrix[actual][predicted]++;
        totals[actual]++;
    }

    // Display the confusion matrix as counts
    printf("Confusion Matrix:\n");
    printf("%10s", "");
    for (int i = 0; i < num_classes; i++) {
        printf("%10d", i + 1); // Print class labels as headers
    }
    printf("\n");

    for (int i = 0; i < num_classes; i++) {
        printf("%10d", i + 1); // Print class labels on the left
        for (int j = 0; j < num_classes; j++) {
            printf("%10d", matrix[i][j]); // Print the count
        }
        printf("\n");
    }

    // Clean up
    for (int i = 0; i < num_classes; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(totals);
}


void showHistogram_poza(float** procente, char* path) {
    int histSize = 256;
    Mat img = imread(path);
    if (img.empty()) {
        printf("Nu am putut deschide imaginea\n");
        (*procente)[0] = 0.0f;
        (*procente)[1] = 0.0f;
        (*procente)[2] = 0.0f;
        return;
    }

    int countRed = 0;
    int countGreen = 0;
    int countBlue = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            if (pixel[1] > pixel[0] && pixel[1] > pixel[2] && pixel[1] > 100) {
                countGreen++;
            }
            if (pixel[2] > pixel[0] && pixel[2] > pixel[1] && pixel[2] > 100) {
                countRed++;
            }
            if (pixel[0] > pixel[1] && pixel[0] > pixel[2] && pixel[0] > 100) {
                countBlue++;
            }
        }
    }

    float total_pixels = img.rows * img.cols;

    (*procente)[0] = (countBlue / total_pixels) * 100;
    (*procente)[1] = (countGreen / total_pixels) * 100;
    (*procente)[2] = (countRed / total_pixels) * 100;

}

int* afisare_etichete_smart(Train_Element* original, int size_list) {
    srand(time(NULL));

    //facem un nou vector pentru noile etichete generate random
    int* etichete_generate = NULL;
    int size_etichete_generate = size_list;
    etichete_generate = (int*)malloc(size_etichete_generate * sizeof(int));
    const char* copied_path = "E:\\An3\\IP\\proiect\\train-scene classification\\train\\";
    int lungime_path = strlen(copied_path);
    float* procentaje = NULL;
    procentaje = (float*)calloc(3, sizeof(float));

    for (int i = 0; i < size_etichete_generate; i++) {
        //generare eticheta random 1-6
        char* poza = original[i].name;
        int lungime_poza = strlen(poza) + lungime_path + 1;
        char* path_poza = new char[lungime_poza];
        strcpy(path_poza, copied_path);
        strcat(path_poza, poza);
        printf("%s\n", path_poza);
        showHistogram_poza(&procentaje, path_poza);
        int random_number = 1 + rand() % (2 - 1 + 1);
        int culoare = -1;
        float maxim = -1.0f;
        for (int j = 0; j < 3; j++) {
            if (procentaje[j] > maxim) {
                culoare = j;
                maxim = procentaje[j];
            }
        }
        if (culoare == 0) {
            if (random_number == 1) {
                etichete_generate[i] = 4;
            }
            else {
                etichete_generate[i] = 6;
            }
        }
        else {
            if (culoare == 1) {
                if (random_number == 1) {
                    etichete_generate[i] = 2;
                }
                else {
                    etichete_generate[i] = 3;
                }
            }
            else {
                if (random_number == 1) {
                    etichete_generate[i] = 1;
                }
                else {
                    etichete_generate[i] = 5;
                }
            }
        }
        printf("Albastru:%.2f%%,Verde:%.2f%%,Rosu:%.2f%% --Eticheta:%d\n", procentaje[0], procentaje[1], procentaje[2], etichete_generate[i]);
    }
    free(procentaje);
    afisare_acuratete(original, etichete_generate, size_etichete_generate);
    return etichete_generate;
}

int clasificaScena(float* procentaje) {
    int eticheta = 0;
    const float prag_albastru = 30.0;
    const float prag_verde = 50.0;
    const float prag_rosu = 20.0;

    float suma = procentaje[0] + procentaje[1] + procentaje[2];

    float procent_albastru = procentaje[0] / suma * 100;
    float procent_verde = procentaje[1] / suma * 100;
    float procent_rosu = procentaje[2] / suma * 100;

    if (procent_albastru > prag_albastru) {
        eticheta = (procent_verde > 20) ? 4 : 6;
    }
    else if (procent_verde > prag_verde) {
        eticheta = 2;
    }
    else if (procent_rosu > prag_rosu) {
        eticheta = (procent_verde > 15) ? 1 : 5;
    }
    else {
        eticheta = 1;
    }
    return eticheta;
}

int clasificaScena2(float* procente) {
    float score = 0;
    float maxScore = 0;
    int eticheta = 0;

    score = procente[0] * 1.2 + procente[2] * 0.5;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 6;
    }
    score = procente[1] * 1.5;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 2;
    }
    score = procente[0] * 1.1 + procente[1] * 0.8 + procente[2] * 0.3;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 4;
    }
    score = procente[1] * 1.0 + procente[2] * 1.0;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 3;
    }
    score = procente[2] * 1.3;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 1;
    }
    score = procente[2] * 0.8 + procente[0] * 0.7;
    if (score > maxScore) {
        maxScore = score;
        eticheta = 5;
    }
    return eticheta;
}

int* afisare_etichete_smart2(Train_Element* original, int size_list) {
    srand(time(NULL));

    //facem un nou vector pentru noile etichete generate random
    int* etichete_generate = NULL;
    int size_etichete_generate = size_list;
    etichete_generate = (int*)malloc(size_etichete_generate * sizeof(int));
    const char* copied_path = "E:\\An3\\IP\\proiect\\train-scene classification\\train\\";
    int lungime_path = strlen(copied_path);
    float* procentaje = NULL;
    procentaje = (float*)calloc(3, sizeof(float));
    for (int i = 0; i < size_etichete_generate; i++) {
        //generare eticheta random 1-6
        char* poza = original[i].name;
        int lungime_poza = strlen(poza) + lungime_path + 1;
        char* path_poza = new char[lungime_poza];
        strcpy(path_poza, copied_path);
        strcat(path_poza, poza);
        printf("%s\n", path_poza);
        showHistogram_poza(&procentaje, path_poza);
        int random_number = 1 + rand() % (2 - 1 + 1);
        int culoare = -1;
        float maxim = -1.0f;
        for (int j = 0; j < 3; j++) {
            if (procentaje[j] > maxim) {
                culoare = j;
                maxim = procentaje[j];
            }
        }
        etichete_generate[i] = clasificaScena2(procentaje);
        printf("Albastru:%.2f%%,Verde:%.2f%%,Rosu:%.2f%% --Eticheta:%d\n", procentaje[0], procentaje[1], procentaje[2], etichete_generate[i]);
    }
    free(procentaje);
    return etichete_generate;
}




void show_smart_generated_labels(Train_Element* new_test_list, int* etichete_generate_fourier, int new_test_size) {
    printf("Smart Generated Labels for Test Data:\n");
    for (int i = 0; i < new_test_size; i++) {
        printf("Image: %s, Fourier Label: %d\n", new_test_list[i].name, etichete_generate_fourier[i]);
    }
    afisare_acuratete(new_test_list, etichete_generate_fourier, new_test_size);
}

void computeFourierFeatures(const Mat& src, std::vector<float>& features)
{
    // Expand input image to optimal size
    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

    // Make place for both the complex and real values
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    // Perform the Discrete Fourier Transform
    dft(complexI, complexI);

    // Compute the magnitude
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];

    // Crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // Apply log transformation to the magnitude spectrum
    magI += Scalar::all(1); // Add 1 to shift values to avoid log(0)
    log(magI, magI);

    // Rearrange the quadrants of Fourier image so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // Compute mean and standard deviation of the magnitude spectrum
    Scalar mean, stddev;
    meanStdDev(magI, mean, stddev);

    // Standardize the values (Z-score normalization)
    float mean_val = static_cast<float>(mean[0]);
    float stddev_val = static_cast<float>(stddev[0]);

    features.push_back(mean_val);
    features.push_back(stddev_val);

    
}

int classifySceneBasedOnFourier(const std::vector<float>& features)
{
    float mean = features[0];
    float stddev = features[1];

   

    // Adjust thresholds based on normalized values
    if (mean < 6.0 )
        return 1; // Buildings
    else if (mean < 6.5)
        return 2; // Forests
    else if (mean < 7.0)
        return 3; // Mountains
    else if (mean < 7.5)
        return 4; // Glacier
    else if (mean < 8.0)
        return 5; // Street
    else
        return 6; // Sea
}

int* classifyUsingFourier(Train_Element* original, int size_list, const char* folder_path)
{
    int* etichete_generate = (int*)malloc(size_list * sizeof(int));

    for (int i = 0; i < size_list; i++)
    {
        char full_path[MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s\\%s", folder_path, original[i].name);
        printf("Attempting to read image from path: %s\n", full_path); // Debugging line
        Mat src = imread(full_path, IMREAD_GRAYSCALE);

        if (src.empty())
        {
            printf("Could not open or find the image: %s\n", full_path);
            etichete_generate[i] = -1; // invalid label
            continue;
        }

        std::vector<float> features;
        computeFourierFeatures(src, features);
        etichete_generate[i] = classifySceneBasedOnFourier(features);

        printf("Image: %s, Mean: %.2f, StdDev: %.2f, Label: %d\n",
            original[i].name, features[0], features[1], etichete_generate[i]);
    }

    return etichete_generate;
}

// LBP Feature Extraction
void computeLBP(const Mat& src, Mat& lbp) {
    lbp = Mat::zeros(src.size(), CV_8UC1);

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i, j);
            uchar code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i, j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i, j - 1) > center) << 0;
            lbp.at<uchar>(i, j) = code;
        }
    }
}

void computeLBPHistogram(const Mat& lbp, vector<float>& hist) {
    hist.resize(256, 0);

    for (int i = 0; i < lbp.rows; i++) {
        for (int j = 0; j < lbp.cols; j++) {
            int bin = lbp.at<uchar>(i, j);
            hist[bin]++;
        }
    }

    for (size_t i = 0; i < hist.size(); i++) {
        hist[i] /= static_cast<float>(lbp.total());
    }
}

// HOG Feature Extraction
void computeHOG(const Mat& src, vector<float>& descriptors) {
    HOGDescriptor hog;
    hog.winSize = Size(64, 64); // Set window size

    // Compute HOG descriptors
    vector<Point> locations;
    hog.compute(src, descriptors, Size(8, 8), Size(0, 0), locations);
}

int classifyUsingLBPHOG(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = src;
    }

    // Compute LBP
    Mat lbp;
    computeLBP(gray, lbp);

    // Compute LBP Histogram
    vector<float> lbpHist;
    computeLBPHistogram(lbp, lbpHist);

    // Compute HOG
    vector<float> hogDescriptors;
    computeHOG(gray, hogDescriptors);

    // Combine LBP and HOG features for classification
    vector<float> features;
    features.insert(features.end(), lbpHist.begin(), lbpHist.end());
    features.insert(features.end(), hogDescriptors.begin(), hogDescriptors.end());

    // Here we use a simple threshold-based approach, but you can train a machine learning model
    // such as SVM, Random Forest, etc., using these features for better accuracy.
    // For demonstration, we use simple rules:
    float lbpSum = std::accumulate(lbpHist.begin(), lbpHist.end(), 0.0f);
    float hogSum = std::accumulate(hogDescriptors.begin(), hogDescriptors.end(), 0.0f);

    if (lbpSum > 1.0 && hogSum > 1.0) return 1; // Example threshold for buildings
    else if (lbpSum > 0.8 && hogSum > 0.8) return 2; // Example threshold for forests
    else if (lbpSum > 0.6 && hogSum > 0.6) return 3; // Example threshold for mountains
    else if (lbpSum > 0.4 && hogSum > 0.4) return 4; // Example threshold for glaciers
    else if (lbpSum > 0.2 && hogSum > 0.2) return 5; // Example threshold for streets
    else return 6; // Sea or other
}

int main()
{
    char* path = get_path();
    char* copied_path = (char*)malloc(sizeof(char) * (strlen(path) + 1));
    strcpy(copied_path, path);
    int op;
    char buffer[100];

    // Citim CSV-ul original cu train si impartim 50/50 in alte 2 liste: new_train si new_test
    int size_train;
    Train_Element* train_list = read_train(copied_path, &size_train);

    Train_Element* new_train_list = NULL;
    Train_Element* new_test_list = NULL;

    int new_train_size = size_train / 2;
    int new_test_size = size_train - new_train_size;

    new_train_list = (Train_Element*)malloc(new_train_size * sizeof(Train_Element));
    if (!new_train_list) {
        printf("Memory allocation failed for the first list\n");
        exit(1);
    }

    new_test_list = (Train_Element*)malloc(new_test_size * sizeof(Train_Element));
    if (!new_test_list) {
        printf("Memory allocation failed for the second list\n");
        exit(1);
    }

    for (int i = 0; i < new_train_size; i++) {
        new_train_list[i] = train_list[i];
    }

    for (int i = 0; i < new_test_size; i++) {
        new_test_list[i] = train_list[i + new_train_size];
    }

    // array cu etichetele generate random pentru test 
    int* etichete_generate = generare_etichete(new_test_size);

    do
    {
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - Basic image opening...\n");
        printf(" 2 - Open BMP images from folder\n");
        printf(" 3 - Color to Gray\n");
        printf(" 4 - Show train and test lists\n");
        printf(" 5 - Show smart generated labels(RGB)\n");
        printf(" 6 - Show acc\n");
        printf(" 7 - Show confusion matrix\n");
        printf(" 8 - Show smart generated labels(random)\n");
        printf(" 9 - Fourier Transform Classification\n");
        printf("10 - LBP and HOG Classification\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        fgets(buffer, sizeof(buffer), stdin);
        sscanf(buffer, "%d", &op);
        switch (op)
        {
        case 1:
            testOpenImage();
            break;
        case 2:
            testOpenImagesFld();
            break;
        case 3:
            testColor2Gray();
            break;
        case 4:
            show_split_train(new_train_list, new_test_list, new_train_size, new_test_size);
            break;
        case 5:
            afisare_etichete_smart(new_test_list, new_test_size);
            break;
        case 6:
            afisare_acuratete(new_test_list, etichete_generate, new_test_size);
            break;
        case 7:
            showConfusionMatrix(new_test_list, etichete_generate, new_test_size, 6);
            break;
        case 8:
            afisare_etichete(new_test_list, new_test_size);
            break;
        case 9:
        {
            int* etichete_generate_fourier = classifyUsingFourier(new_test_list, new_test_size, "E:\\An3\\IP\\proiect\\train-scene classification\\train");
            show_smart_generated_labels(new_test_list, etichete_generate_fourier, new_test_size);
            free(etichete_generate_fourier);
        }
        break;
        case 10:
        {
            int size = new_test_size;
            int* label = (int*)malloc(size * sizeof(int));

            for (int i = 0; i < new_test_size; i++) {
                char full_path[MAX_PATH];
                snprintf(full_path, sizeof(full_path), "E:\\An3\\IP\\proiect\\train-scene classification\\train\\%s", new_test_list[i].name);
                Mat src = imread(full_path);
                if (src.empty()) {
                    printf("Could not open or find the image: %s\n", full_path);
                    continue;
                }
                label[i] = classifyUsingLBPHOG(src);
                printf("Image: %s, LBP-HOG Label: %d\n", new_test_list[i].name, label[i]);
            }
            afisare_acuratete(new_test_list, label, new_test_size);
        }
        break;
        }
    } while (op != 0);

    free(train_list);
    free(new_train_list);
    free(new_test_list);
    free(etichete_generate);
    free(copied_path);

    return 0;
}
