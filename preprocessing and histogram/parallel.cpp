#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <english_stem.h>
#include <codecvt>
#include <locale>
#include <chrono>
#include <cuda_runtime.h>
#include "Histogram.h"

#define SIZE    (100*1024*1024)

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

std::string to_string(std::wstring wstr)
{
    return strconverter.to_bytes(wstr);
}

std::wstring to_wstring(std::string str)
{
    return strconverter.from_bytes(str);
}

void call_kernel(int* values, int length, int* hist, int hist_length);

typedef boost::tokenizer<boost::char_separator<wchar_t>,
std::wstring::const_iterator, std::wstring> tokenizer;


int main() {
    int *values, *hist; int length, hist_length;
    scanf("%d", &length);
    printf("\n");
    scanf("%d", &hist_length);
    values = (int*) malloc(sizeof(int) * length);
    hist = (int*) malloc(sizeof(int) * hist_length);

    std::string file_name;
    std::cin >> file_name;
    std::wifstream inputFile("corpus/" + file_name); //"corpus/verylarge_corpora.txt.final");
    if (!inputFile) {
        std::cerr << "Failed to open input file." << std::endl;
        return 1;
    }
    auto start = high_resolution_clock::now();
    std::unordered_map<std::wstring, int> wordIds;
    int currentId = 0, j = 0;

    std::wstring line;
    boost::char_separator<wchar_t> sep(L" \t\n.,;:!?'\"()[]{}<>");

    stemming::english_stem<> stemmer;
    // int len_max = 0;
    while (std::getline(inputFile, line)) {
        //typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        tokenizer tokens(line, sep);
        for (const auto& token : tokens) {
            // Convert token to lowercase
            std::wstring lowerToken = token;
            std::transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(), ::tolower);
            // Remove punctuation
            lowerToken.erase(std::remove_if(lowerToken.begin(), lowerToken.end(), ::ispunct), lowerToken.end());
            // Skip stop words
            if (stopWords.count(lowerToken) > 0) {
                continue;
            }
            stemmer(lowerToken);
            // Assign unique ID to each word
            if (wordIds.find(lowerToken) == wordIds.end()) {
                wordIds[lowerToken] = currentId++;
                values[j++] = currentId;
            } else {
                values[j++] = wordIds[lowerToken];
            }

            if (j == length) {
                //histogramHost(values, length, hist, hist_length);
                j = 0;
            }
            // std::cout << to_string(lowerToken) << '\n';
            // len_max = len_max > lowerToken.size() ? len_max : lowerToken.size();

            //if (wordIds.find(lowerToken) != wordIds.end()) {
            //    wordIds[lowerToken] += 1;
            //}
            //std::cout << "Word: " << to_string(lowerToken) << ", ID: " << wordIds[lowerToken] << std::endl;
        }
    }
    if (j < 0) {
        call_kernel(values, length, hist, hist_length);
    }

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
    printf("elapsed time: %ld\n", 
            duration_cast<microseconds>(stop - start).count());
    //std::cout << "Most frequent Word: " << to_string(most_freq_word) <<
    //    ", frequency: " << max_freq << std::endl;
    //return (int)duration_cast<milliseconds>(stop - start).count();
}


