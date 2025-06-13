#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include <english_stem.h>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <codecvt>
#include <locale>
#include <chrono>
#include <math.h>
#include <omp.h>

#include "book.h"

#define SIZE    (100*1024*1024)
#define MAX_THREADS_PER_SM 1536
#define SM_COUNT 14

int no_streams = 4;

std::unordered_set<std::wstring> stopWords = {
    L"i", L"me", L"my", L"myself", L"we", L"our", L"ours", L"ourselves", L"you",
    L"your", L"yours", L"yourself", L"yourselves", L"he", L"him", L"his", L"himself",
    L"she", L"her", L"hers", L"herself", L"it", L"its", L"itself", L"they", L"them",
    L"their", L"theirs", L"themselves", L"what", L"which", L"who", L"whom", L"this",
    L"that", L"these", L"those", L"am", L"is", L"are", L"was", L"were", L"be", L"been",
    L"being", L"have", L"has", L"had", L"having", L"do", L"does", L"did", L"doing", L"a",
    L"because", L"as", L"until", L"while", L"of", L"at", L"by", L"for", L"with",
    L"about", L"against", L"between", L"into", L"through", L"during", L"before",
    L"after", L"above", L"below", L"to", L"from", L"up", L"down", L"in", L"out",
    L"on", L"off", L"over", L"under", L"again", L"further", L"then", L"once",
    L"here", L"there", L"when", L"where", L"why", L"how", L"all", L"any", L"both", L"",
    L"each", L"few", L"more", L"most", L"other", L"some", L"such", L"no", L"nor",
    L"not", L"only", L"own", L"same", L"so", L"than", L"too", L"very", L"s", L"t", L"can",
    L"will", L"just", L"don", L"should", L"now", L"an", L"the", L"and", L"but", L"if", L"or"
};

using namespace std::chrono;
using convert_t = std::codecvt_utf8<wchar_t>;

std::wstring_convert<convert_t, wchar_t> strconverter;
typedef boost::tokenizer<boost::char_separator<wchar_t>,
std::wstring::const_iterator, std::wstring> tokenizer;

inline std::string to_string(std::wstring wstr) { return strconverter.to_bytes(wstr); }
inline std::wstring to_wstring(std::string str) { return strconverter.from_bytes(str); }
void histogramHost(const int* values, int length, int* hist, int hist_length);
int hist_serial(const int* A, int width, int* hist, int hist_width);


int main(int argc, char* argv[]){
    int *values, *hist; int length, hist_length, is_parallel;
    std::string file_name;
	if (argc < 6) {
		printf("USAGE: fragment_passed_to_kernel_length hist_length no_streams corp_name is_parallel\n");
		exit(1);
	}
	check_gpu_availability();
	length      = std::stoi(argv[1]);
	hist_length = std::stoi(argv[2]);
	no_streams  = std::stoi(argv[3]);
    is_parallel = std::stoi(argv[5]);
    file_name   = argv[4];
    values      = (int*) malloc(sizeof(int) * length);
    hist        = (int*) malloc(sizeof(int) * hist_length);
    // std::cin >> file_name;
    std::wifstream inputFile("corpus/" + file_name); 
    if (!inputFile) {
        std::cerr << "Failed to open input file." << std::endl;
        return 1;
    }
    auto start = high_resolution_clock::now();
    int currentId = 0, j = 0;
    std::unordered_map<std::wstring, int> wordIds;
    std::wstring line;
    boost::char_separator<wchar_t> sep(L" \t\n.,;:!?'\"()[]{}<>");
    stemming::english_stem<> stemmer;
    auto search = wordIds.find(L"");
    #pragma omp parallel num_threads(1 + is_parallel * 3) 
    {
    #pragma omp single 
    {
    while (std::getline(inputFile, line)) {
        //typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        tokenizer tokens(line, sep);
        for (const auto& token : tokens) {
            // Convert token to lowercase
            std::wstring lowerToken = token;
            //#pragma omp task
            //{
            std::transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(), ::tolower);
            // Remove punctuation
            lowerToken.erase(std::remove_if(lowerToken.begin(), lowerToken.end(), ::ispunct), lowerToken.end());
            // Skip stop words
            //}
            //#pragma omp task
            //{
            stemmer(lowerToken);
            // Assign unique ID to each word
            search = wordIds.find(lowerToken); 
            if (search != wordIds.end()) {
                values[j++] = search -> second;
            } else {
                values[j++] = currentId;
                wordIds[lowerToken] = currentId++;
            }
            //}
            #pragma omp taskwait
            if (stopWords.count(lowerToken) > 0) {continue;}
            if (j == length) {
                j = 0;
                histogramHost(values, length, hist, hist_length);
            }
            // std::cout << to_string(lowerToken) << '\n';
            // len_max = len_max > lowerToken.size() ? len_max : lowerToken.size();

            //if (wordIds.find(lowerToken) != wordIds.end()) {
            //    wordIds[lowerToken] += 1;
            //}
            //std::cout << "Word: " << to_string(lowerToken) << ", ID: " << wordIds[lowerToken] << std::endl;
        }
    }
    }
    }
    if (j < 0) 
        histogramHost(values, length, hist, hist_length);
    //hist_serial(values, length, hist, hist_length);
    //int max_freq = 0;
    //std::wstring most_freq_word;
    //for(auto it=wordIds.begin();it!=wordIds.end();it++) {
    //    //std::cout<<to_string(it->first) << ", count: " << it->second << "\n"; 
    //    if (max_freq < it->second) {
    //        max_freq = it->second;
    //        most_freq_word = it->first;
    //    }
    //}
    inputFile.close();
    auto stop = high_resolution_clock::now();
    printf("elapsed time: %lld\n", 
            duration_cast<microseconds>(stop - start).count());
    //std::cout << "Most frequent Word: " << to_string(most_freq_word) <<
    //    ", frequency: " << max_freq << std::endl;
    //return (int)duration_cast<milliseconds>(stop - start).count();
}

//kernel for computing histogram right in memory
__global__ void hist_inGlobal (const int* values, int width, int* hist) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	while(idx < width){
		atomicAdd(&hist[values[idx]], 1);
		idx += stride;
	}
}

int hist_serial(const int* A, int width, int* hist, int hist_width){
	for (int i = 0; i < width; i++)
		hist[A[i]]++;
    return 0;
}

void histogramHost(const int* values, int width, int* hist,
					 int hist_width){
	int const magic_number = 268435456;
	int *dev_values, *dev_hist;
	int size_dvalues = std::min(width, magic_number) * sizeof(int);
	int size_hist = hist_width * sizeof(int);
	checkCudaErrors(cudaMalloc(&dev_values, size_dvalues));
	checkCudaErrors(cudaMalloc(&dev_hist, size_hist));
	checkCudaErrors(cudaMemcpy(dev_hist, hist, size_hist, cudaMemcpyHostToDevice));
	dim3 blocks(std::min(512, hist_width),1,1);
	dim3 grid(MAX_THREADS_PER_SM/blocks.x*std::floor(SM_COUNT/no_streams), 1, 1); 
	std::vector<cudaStream_t> streams(no_streams); 
	for (int i = 0; i < no_streams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])); 
	int stream_size = size_dvalues / no_streams / sizeof(int);
	for (int i = 0; i < width; i += magic_number) {
		const int* head = values + i;
		for (int j = 0; j < no_streams; j++)
		{
			checkCudaErrors(cudaMemcpyAsync(
							dev_values + (j * stream_size),
							head + (j * stream_size),
							stream_size * sizeof(int),
							cudaMemcpyHostToDevice, streams[j]
							));
			hist_inGlobal<<<grid, blocks, 0, streams[j]>>>
					(dev_values + j * stream_size,stream_size, dev_hist);
		}
	}
    #pragma omp task
    {
	cudaDeviceSynchronize();
	cudaCheckLastError(__FILE__, __LINE__);
	for (int i = 0; i < no_streams; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i])); 
	checkCudaErrors(cudaMemcpy(hist,dev_hist,size_hist,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dev_hist));
	checkCudaErrors(cudaFree(dev_values));
    }
}
