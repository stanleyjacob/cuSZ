//
// Created by jtian on 4/24/20.
//

template <typename Huff>
inline int get_symlen(Huff sym)
{
    return (int)*((uint8_t*)&sym + sizeof(Huff) - 1);
}

template <typename Input, typename Dict>
std::tuple<double, double> get_avgbw_entropy(Input* q, unsigned int len, Dict* cb, unsigned int cb_len)
{
    auto d_q_hist = mem::CreateDeviceSpaceAndMemcpyFromHost(q, len);
    auto d_freq   = mem::CreateCUDASpace<unsigned int>(cb_len);
    wrapper::GetFrequency(d_q_hist, len, d_freq, cb_len);
    auto freq = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, cb_len);

    double avg_bw = 0, entropy = 0;
    for (auto i = 0; i < cb_len; i++) {
        auto   bw = get_symlen(cb[i]);
        auto   f  = freq[i] * 1.0;
        double p  = f * 1.0 / len;
        if (bw != 255) avg_bw += f * bw;
        if (bw != 255) entropy += p * log(p);
    }
    avg_bw /= len;
    entropy = -entropy;

    cout << log_info << "average bw:\t" << avg_bw << endl;
    cout << log_info << "entropy:\t" << entropy << endl;

    cudaFree(d_q_hist), cudaFree(d_freq);

    return {avg_bw, entropy};
}

template <typename Input, typename Dict>
void filter_out(Input* q, uint32_t len, Dict* cb, uint32_t cb_len, uint32_t threshold_bw = 5)  // prepare for extra
                                                                                               // outliers
{
    // find shortest "special" symbol
    unsigned int shortest = 255;
    unsigned int count    = 0;
    Input        special;
    Dict         special_code;
    for (auto i = 0; i < cb_len; i++) {
        //        cout << i << "\t" << get_symlen(cb[i]) << "\t" << bitset<32>(cb[i]) << endl;

        auto sym_len = get_symlen(cb[i]);
        if (sym_len < shortest) {
            shortest     = sym_len;
            special      = i;
            special_code = cb[i];
        }
    }
    cout << log_dbg << "shortest codeword len\t" << shortest << "\tcodeword\t" << bitset<32>(special_code) << endl;
    cout << log_dbg << "filtering threshold bw\t" << threshold_bw << endl;

    for (auto i = 0; i < len; i++) {
        auto sym     = cb[q[i]];
        auto sym_len = get_symlen(sym);
        if (sym_len > threshold_bw) {
            q[i] = special;
            count++;
        }
    }
    cout << log_info << count << " are changed, " << (count * 100.0 / len) << "%" << endl;
}