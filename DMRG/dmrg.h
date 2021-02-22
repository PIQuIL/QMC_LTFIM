#ifndef DMRG_H
#define DMRG_H

#include "itensor/all.h"
#include <cstdlib>
#include <random>
#include <fstream>

using namespace std;
using namespace itensor;

class DMRG
{

    int N_;
    MPO Hamiltonian_;
    SiteSet sites_;
    MPS psi_;
    double energy_;
    double magnetization_;
    MPS psi0_;

public:
    DMRG(int N) : N_(N) {}

    inline MPS GetPsi()
    {
        return psi_;
    }

    inline double GetEnergy()
    {
        return energy_;
    }

    inline double GetM()
    {
        return magnetization_;
    }

    inline SiteSet GetSiteSet()
    {
        return sites_;
    }

    void Rydberg(double Rb, double delta, double Omega, int trunc)
    {
        vector<vector<double>> Vij;

        // inefficient way to store Vij since a lot of repitition
        for (int i = 1; i < N_; ++i)
        {
            vector<double> tmp;
            
            for (int j = i + 1; j <= N_; ++j)
            {
                auto vij = 1/pow(Rb*(j-i), 6.0);
                tmp.push_back(vij);
            }
            
            Vij.push_back(tmp); 
        }

        sites_ = SpinHalf(N_, {"ConserveQNs=", false});
        auto ampo = AutoMPO(sites_);

        for (int i = 1; i < N_; ++i)
        {
            for (int j = i + 1; j <= N_; ++j)
            {
                if (abs(j-i) <= trunc)
                {
                    ampo += -Vij[i-1][j-i-1] * 4.0, "Sz", i, "Sz", j;
                    ampo += -Vij[i-1][j-i-1] * 2.0, "Sz", i;
                    ampo += -Vij[i-1][j-i-1] * 2.0, "Sz", j;
                }
            }

            ampo += -delta * 2.0, "Sz", i;
            ampo += -Omega * 2.0, "Sx", i;
        }

        ampo += -delta * 2.0, "Sz", N_;
        ampo += -Omega * 2.0, "Sx", N_;

        Hamiltonian_ = toMPO(ampo); 
    }

    void LTFIM1D(double J, double h, double Omega, string BC)
    {
        sites_ = SpinHalf(N_, {"ConserveQNs=", false});
        auto ampo = AutoMPO(sites_);

        for (int j = 1; j < N_; ++j)
        {
            ampo += -J * 4.0, "Sz", j, "Sz", j + 1;
            ampo += -h * 2.0, "Sx", j;
            ampo += -Omega * 2.0, "Sz", j;
        }
        ampo += -h * 2.0, "Sx", N_;
        ampo += -Omega * 2.0, "Sz", N_;

        if (BC == "PBC")
        {
            ampo += -J * 4.0, "Sz", 1, "Sz", N_;
        }

        Hamiltonian_ = toMPO(ampo);
    }

    void InitializeState(string state_type)
    {
        // state_type options:
        // Neel, all up / down, random

        auto state = InitState(sites_);

        if (state_type == "neel_Sz=0")
        {
            for (int i = 1; i <= N_; ++i)
            {
                if (i % 2 == 1)
                    state.set(i, "Up");
                else
                    state.set(i, "Dn");
            }
                
        }

        if (state_type == "neel")
        {
            for (int i = 1; i <= N_; ++i)
            {
                if (i % 2 == 1)
                    state.set(i, "Up");
                else
                    state.set(i, "Dn");
            }
        }

        if (state_type == "random")
        {
            uniform_int_distribution<int> distribution(0, 1);
            mt19937 rd;

            for (int i = 1; i <= N_; ++i)
            {
                auto rng = distribution(rd);

                if (rng == 1)
                    state.set(i, "Up");
                else
                    state.set(i, "Dn");
            }
        }

        if (state_type == "Up")
        {
            for (int i = 1; i <= N_; ++i)
            {
                state.set(i, "Up");
            }
        }

        if (state_type == "Down")
        {
            for (int i = 1; i <= N_; ++i)
            {
                state.set(i, "Dn");
            }
        }

        psi0_ = MPS(state);
    }

    ITensor ContractMPS(MPS psi)
    {
        ITensor psi_contract = psi.A(1);
        for (int i = 2; i <= N_; i++)
        {
            psi_contract *= psi.A(i);
        }
        return psi_contract;
    }

    void Run()
    {
        auto sweeps = Sweeps(100);
        sweeps.maxdim() = 50, 50, 100, 100, 200;

        auto [energy, psi] = dmrg(Hamiltonian_, psi0_, sweeps, "Quiet");
        psi_ = psi;
        
        // measure magnetization
        double Sz;
        double Szj;

        Sz = 0.0;
        for (int j = 1; j <= N_; ++j) 
        {
            psi.position(j);
            Szj = elt(psi.A(j) * sites_.op("Sz", j) * dag(prime(psi.A(j), "Site")));
            Sz += Szj;
        }

        magnetization_ = Sz / float(N_);
        // mult by 2 since itensor works with +/- 1/2 and not +/- 1
        energy_ = energy / float(N_);
        printfln("\nGround State E = %.10f", energy_);
        printfln("\nGround State M = %.10f", magnetization_);
    }
};

#endif
