// 20-09-07

#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

using namespace std;

template <typename T>
T* bernoulli_gen(size_t length, double true_prob)
{
    std::random_device          rd;
    std::mt19937                gen(rd());
    std::bernoulli_distribution ber(true_prob);  // give "true" true_prob of the time

    auto count = 0;
    auto d     = new T[length];
    for (auto i = 0; i < length; i++) {
        if (ber(gen)) {
            d[i] = 1.14;
            count++;
        }
    }
    cout << "count: " << count << ", sparsity ratio (expected, real): (" << true_prob << ", " << count / (length * 1.0) << ")" << endl;
    return d;
}