#include "itensor/all.h"
#include <cstdlib>
#include <sstream>
#include <random>
#include <fstream>

#include <string>
#include <iostream>
#include <vector>
#include <iomanip>

#include "dmrg.h"
#include "sampler.h"

using namespace std;
using namespace itensor;

int main(int argc, char* argv[]) {

    int N = atoi(argv[1]);
    double Rb = atof(argv[2]);
    double delta = atof(argv[3]);
    double Omega = atof(argv[4]);
    int trunc = atoi(argv[5]);
    int num_samples = atoi(argv[6]);
    
    // run DMRG
    DMRG dmrg_(N);
    dmrg_.Rydberg(Rb, delta, Omega, trunc);
    dmrg_.InitializeState("random");
    dmrg_.Run();
    auto energy = dmrg_.GetEnergy();
    auto magnetization = dmrg_.GetAbsM();

    // sample the DMRG wavefunction
    MPS psi;
    psi = dmrg_.GetPsi();
    SiteSet sites = dmrg_.GetSiteSet();

    stringstream sample_path_;
    stringstream sample_bases_path_;
    sample_path_ << "samples/DMRG_samples_N=" << N << "_Rb=" << Rb << "_delta=" << delta << "_Omega=" << Omega << "_trunc=" << trunc << ends;
    auto sample_path = sample_path_.str();

    Sampler sampler(N, psi, sites, num_samples, sample_path);
    sampler.Sample();

    // save energies
    stringstream DMRG_path;
    DMRG_path << "observables/DMRG_observables_N=" << N << "_Rb=" << Rb << "_delta=" << delta << "_Omega=" << Omega << "_trunc=" << trunc << ends;
    ofstream DMRG_file(DMRG_path.str());

    DMRG_file << "E0/N\t" << "|M|/N" << endl;
    DMRG_file << energy << "\t" << magnetization << endl;

}
