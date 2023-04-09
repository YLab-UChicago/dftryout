#include <iostream>
#include <ctime>
#include "resnet20.hh"
#include "stationaries.cpp"


using namespace std;

int main(int argc, char const *argv[])
{
    /* initialize res20 */
    // ResNet20 res20;
    const int input_size = 10;
    const int filter_size = 5;
    const int batch_size = 2;
    const int height = 5;
    const int width = 5;
    const int depth = 1;
    const int num_filters = 1;
    const int filter_height = 3;
    const int filter_width = 3;
    const int padding = 0;
    const int strides = 1;
    const int output_size = (height - filter_height + 2 * padding) / strides + 1;
    float inputs[input_size] = {1.5, 2.5, 3.33, 4.31, 5.1231, 6.123, 7.12315, 8.12893, 9.1231, 10.123513461345};
    float filters[filter_size] = {1, 2, 3, 4, 5};
    float output[output_size];

    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;

    // running output stationary full precision
    c_start = std::clock();
    conv_os_fp(inputs, filters, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Full precision output CPU time used: " << time_elapsed_ms << " ms\n";


    cout << "output stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output[i] << " ";
    }
    cout << "output stationary tested" << endl;
    
    // running input stationary full precision
    c_start = std::clock();
    conv_is_fp(inputs, filters, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Full precision input CPU time used: " << time_elapsed_ms << " ms\n";


    cout << "input stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output[i] << " ";
    }
    cout << "input stationary tested" << endl;
    

    // running weight stationary full precision
    c_start = std::clock();
    conv_ws_fp(inputs, filters, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Full precision weight stationary CPU time used: " << time_elapsed_ms << " ms\n";


    cout << "weight stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output[i] << " ";
    }
    cout << "weight stationary tested" << endl;
    
    
    // const int input_size = 10;kk
    // const int filter_size = 5;
    // const int batch_size = 2;
    // const int height = 5;
    // const int width = 5;
    // const int depth = 1;
    // const int num_filters = 1;
    // const int filter_height = 3;
    // const int filter_width = 3;
    // const int padding = 0;
    // const int strides = 1;
    // const int output_size = (height - filter_height + 2 * padding) / strides + 1;

    int inputs_bn[input_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int filters_bn[filter_size] = {1, 2, 3, 4, 5};
    int output_bn[output_size];

    c_start = std::clock();
    conv_os_bn(inputs_bn, filters_bn, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output_bn);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    cout << "binary output stationary CPU time used: " << time_elapsed_ms << " ms\n";

    cout << "binary output stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output_bn[i] << " ";
    }
    cout << "binary output stationary finish testing" << endl;

    //binary convolution input stationary
    c_start = std::clock();
    conv_is_bn(inputs_bn, filters_bn, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output_bn);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    cout << "binary input stationary CPU time used: " << time_elapsed_ms << " ms\n";

    cout << "binary input stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output_bn[i] << " ";
    }
    cout << "binary input stationary finish testing" << endl;
    

    //binary convolution weight stationary
    c_start = std::clock();
    conv_ws_bn(inputs_bn, filters_bn, batch_size, height, width, depth, num_filters, filter_height, filter_width, padding, strides, output_bn);
    c_end = std::clock();

    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    cout << "binary weight stationary CPU time used: " << time_elapsed_ms << " ms\n";

    cout << "binary weight stationary start testing" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output_bn[i] << " ";
    }
    cout << "binary weight stationary finish testing" << endl;
    
    return 0;
    
}
