#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <english_stem.h>

int main() {
    std::ifstream inputFile("corpus.txt");
    if (!inputFile) {
        std::cerr << "Failed to open input file." << std::endl;
        return 1;
    }

    std::unordered_map<std::string, int> wordIds;
    std::unordered_set<std::string> stopWords = { "the", "and", "or", "not", "is", "are" };
    int currentId = 0;

    std::string line;
    boost::char_separator<char> sep(" \t\n.,;:!?'\"()[]{}<>");
    while (std::getline(inputFile, line)) {
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        tokenizer tokens(line, sep);
        for (const auto& token : tokens) {
            // Convert token to lowercase
            std::string lowerToken = token;
            std::transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(), ::tolower);

            // Remove punctuation
            lowerToken.erase(std::remove_if(lowerToken.begin(), lowerToken.end(), ::ispunct), lowerToken.end());

            // Skip stop words
            if (stopWords.count(lowerToken) > 0) {
                continue;
            }

            // Assign unique ID to each word
            //if (wordIds.find(lowerToken) == wordIds.end()) {
            //    wordIds[lowerToken] = currentId++;
            //}

            if (wordIds.find(lowerToken) != wordIds.end()) {
                wordIds[lowerToken] += 1;
            } else {
                wordIds[lowerToken] = 1;
            }

            std::cout << "Word: " << lowerToken << ", ID: " << wordIds[lowerToken] << std::endl;
        }
    }

    inputFile.close();

    return 0;
}
