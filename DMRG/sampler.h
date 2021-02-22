#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include "itensor/all.h"
#include <cstdlib>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;
using namespace itensor;

// TODO: bug when rotating wavefunction (not conserving QNs)
// bug in hadamard / kadamard when setting matrix elements 

class Sampler
{

private:
    int N_; // number of sites (currently 1D only)

    int num_samples_; // number of samples to generate in every input basis

    complex<double> I_; // i

    vector<vector<string>> bases_; // bases to generate samples in

    MPS psi_; // the MPS from DMRG calculation

    string sample_path_; // contains generated samples

    SiteSet sites_; // SiteSet from DMRG calculation

public:
    Sampler(int N, MPS &psi, SiteSet sites, int num_samples, string sample_path) : N_(N), psi_(psi), sites_(sites), num_samples_(num_samples), sample_path_(sample_path), I_(0, 1){};

    void Sample()
    {
        // perform sampling
        vector<string> basis;
        ITensor cap;
        vector<int> sample(N_);
        random_device rd;
        mt19937 rng(rd());

        ofstream sample_file(sample_path_);

        LoadOnlyCompBasis();

        for (int b = 0; b < bases_.size(); b++)
        {
            // generate num_samples samples for all bases
            basis = bases_[b];

            MPS psi = RotateMPS(basis);
            psi.position(1);

            for (int j = 0; j < num_samples_; j++)
            {
                for (auto i : range1(N_))
                {
                    auto tensor = i == 1 ? psi.A(i) : cap * psi.A(i);
                    auto rho = prime(dag(tensor), "Site") * tensor;
                    auto I = sites_(i);
                    auto Ip = prime(I);

                    vector<Real> weights(dim(I) + 1);
                    for (auto idx : range1(dim(I)))
                    {
                        weights[idx] = rho.eltC(I(idx), Ip(idx)).real();
                    }

                    discrete_distribution<> d(weights.begin(), weights.end());
                    sample[i - 1] = d(rng);
                    cap = dag(setElt(I(sample[i - 1]))) * tensor;

                } // N

                // write samples and bases to files
                for (int j = 0; j < N_; j++)
                {
                    sample_file << sample[j] - 1 << " ";
                }
                sample_file << endl;
            } // num_samples
        }     // bases
    }

    ITensor Hadamard(const ITensor &A)
    {
        // Diagonalizes pauliX

        auto s = findIndex(A, "Site");
        auto sp = prime(s);

        ITensor hadamard(s, sp);
        hadamard.set(s(1), sp(1), 1.0 / sqrt(2));
        hadamard.set(s(1), sp(2), 1.0 / sqrt(2));
        hadamard.set(s(2), sp(1), 1.0 / sqrt(2));
        hadamard.set(s(2), sp(2), -1.0 / sqrt(2));

        return hadamard;
    }

    ITensor Kadamard(const ITensor &A)
    {
        // Diagonalizes pauliY

        auto s = findIndex(A, "Site");
        auto sp = prime(s);
        ITensor kadamard(s, sp);

        kadamard.set(s(1), sp(1), 1.0 / sqrt(2));
        kadamard.set(s(1), sp(2), 1.0 / sqrt(2));
        kadamard.set(s(2), sp(1), I_ / sqrt(2));
        kadamard.set(s(2), sp(2), -I_ / sqrt(2));

        return kadamard;
    }

    MPS RotateMPS(vector<string> &basis)
    {
        // rotate the MPS to sample in a desired basis

        // placeholder to do the rotations
        auto psi = psi_;
        ITensor rotation;

        for (int j = 1; j <= N_; j++)
        {

            if (basis[j - 1] != "Z")
            {
                psi.position(j);
                auto wf = psi.A(j);
                
                if (basis[j - 1] == "X")
                {
                    rotation = Hadamard(wf);
                }

                if (basis[j - 1] == "Y")
                    rotation = dag(Kadamard(wf));

                wf *= rotation;
                wf.noPrime();
                psi.setA(j, wf);
            }
        }
        return psi;
    }

    void LoadOnlyCompBasis()
    {
        // bases_ will only contain "Z"
        bases_.resize(1, vector<string>(N_));

        for (int j = 0; j < N_; j++)
        {
            bases_[0][j] = "Z";
        }
    }
        

    void LoadBases(ifstream &bases_file)
    {
        // append information from bases_file

        int num_bases;
        bases_file >> num_bases;
        bases_.resize(num_bases, vector<string>(N_));

        for (int i = 0; i < num_bases; i++)
        {
            for (int j = 0; j < N_; j++)
            {
                bases_file >> bases_[i][j];
            }
        }
    }
};

#endif
