#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../deprecated_SDRB.hh"
#include "../type_trait.hh"
#include "../verify.hh"
#include "workflow.hh"

const size_t DICT_SIZE = 1024;

using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    std::string eb_mode, dataset, datum_path;
    int         ndim;
    bool        if_blocking, if_dualquant;
    double      mantissa, exponent;

    if (argc != 9) {
        cout << "./<program> <ndim> <abs|r2r> <mantissa> <exponent> <if chunking> <if dualquant> <dataset> "
                "<datum_path path>"
             << endl;
        cout << "supported dimension and datasets" << endl;
        cout << "\t1D\t./psz1d 1d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <hacc> /path/to/vx.f32" << endl;
        cout << "\t2D\t./psz2d 1d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <cesm> /path/to/CLDHGH_1_1800_3600.f32"
             << endl;
        cout << "\t3D\t./psz3d 1d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <hurricane|nyx|qmc|qmcpre> "
                "/path/to/CLOUDf48.bin.f32"
             << endl;
        exit(0);
    }
    else {
        ndim         = atoi(argv[1]);
        eb_mode      = std::string(argv[2]);
        mantissa     = std::stod(argv[3]);
        exponent     = std::stod(argv[4]);
        if_blocking  = std::string(argv[5]) == "yesblk";
        if_dualquant = std::string(argv[6]) == "dq";
        dataset      = std::string(argv[7]);
        datum_path   = std::string(argv[8]);
    }

    for_each(argv, argv + 8, [](auto i) { cout << i << " "; });
    cout << endl;
    auto eb_config = new config_t(DICT_SIZE, mantissa, exponent);
    auto dims_L16  = InitializeDemoDims(dataset, DICT_SIZE);
    printf("%-20s%s\n", "filename", datum_path.c_str());
    printf("%-20s%lu\n", "filesize", dims_L16[LEN] * sizeof(float));
    if (eb_mode == "r2r") {  // as of C++ 14, string is directly comparable?
        double value_range = GetDatumValueRange<float>(datum_path, dims_L16[LEN]);
        eb_config->ChangeToRelativeMode(value_range);
    }
    eb_config->debug();
    size_t num_outlier = 0;  // for calculating compression ratio

    // TODO export metadata

    // cout << "block size:\t" << BLK << endl;
    auto ebs_L4 = InitializeErrorBoundFamily(eb_config);
    // TODO change int to int16 for quant code
    if (ndim == 1) {
        MetadataTrait<1>::metadata_t* m;
        psz::cx_sim<1, float, int>(m, datum_path, num_outlier, if_dualquant, if_blocking, true);
    }
    if (ndim == 2) {
        MetadataTrait<2>::metadata_t* m;
        psz::cx_sim<2, float, int>(m, datum_path, num_outlier, if_dualquant, if_blocking, true);
    }
    if (ndim == 3) {
        MetadataTrait<3>::metadata_t* m;
        psz::cx_sim<3, float, int>(m, datum_path, num_outlier, if_dualquant, if_blocking, true);
    }
}
