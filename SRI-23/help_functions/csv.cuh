#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>



/* Function to write LQR problem into a csv file when given also soln vector
 *
 *
 *
 *
 */
template <typename T>
void write_csv(const std::string &filename, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs,
               const T *Q_R, const T *q_r, const T *A_B, const T *d, const T *soln)
{
    // Open the CSV file
    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    // Write nhorizon, nstates, and ninputs
    fout << nhorizon << "," << nstates << "," << ninputs;

    // Calculate the dimensions
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t states_s_controls = nstates + ninputs;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Write Q_R
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < cost_step; ++i)
        {
            fout << "," << Q_R[timestep * cost_step + i];
        }
    }
    // Write only Q during last timestep
    for (int i = 0; i < states_sq; ++i)
    {
        fout << "," << Q_R[(nhorizon - 1) * cost_step + i];
    }

    // Write q_r
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < states_s_controls; ++i)
        {
            fout << "," << q_r[timestep * (states_s_controls) + i];
        }
    }
    // write only q for the last timestep
    for (int i = 0; i < nstates; ++i)
    {
        fout << "," << q_r[(nhorizon - 1) * states_s_controls + i];
    }

    // Write A_B
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < dyn_step; ++i)
        {
            fout << "," << A_B[timestep * dyn_step + i];
        }
    }

    // Write d
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {
        for (int i = 0; i < nstates; ++i)
        {
            fout << "," << d[timestep * nstates + i];
        }
    }

    int soln_size = nstates * nhorizon + ((nstates + ninputs) * nhorizon) - ninputs;
    // Write soln
    for (int timestep = 0; timestep < soln_size; ++timestep)
    {
        fout << "," << soln[timestep];
    }

    // Close the file
    fout.close();

    std::cout << "CSV file has been written successfully." << std::endl;
}

/* Function to write LQR problem into a csv file without soln vector
 *
 *
 *
 *
 */
template <typename T>
void write_csv(const std::string &filename, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs,
               const T *Q_R, const T *q_r, const T *A_B, const T *d)
{
    // Open the CSV file
    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    // Write nhorizon, nstates, and ninputs
    fout << nhorizon << "," << nstates << "," << ninputs;

    // Calculate the dimensions
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t states_s_controls = nstates + ninputs;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Write Q_R
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < cost_step; ++i)
        {
            fout << "," << Q_R[timestep * cost_step + i];
        }
    }
    // write only Q during last timestep
    for (int i = 0; i < states_sq; ++i)
    {
        fout << "," << Q_R[(nhorizon - 1) * cost_step + i];
    }

    // Write q_r
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < states_s_controls; ++i)
        {
            fout << "," << q_r[timestep * (states_s_controls) + i];
        }
    }
    // write only q for the last timestep
    for (int i = 0; i < nstates; ++i)
    {
        fout << "," << q_r[(nhorizon - 1) * states_s_controls + i];
    }
    // Write A_B
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < dyn_step; ++i)
        {
            fout << "," << A_B[timestep * dyn_step + i];
        }
    }

    // Write d
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {
        for (int i = 0; i < nstates; ++i)
        {
            fout << "," << d[timestep * nstates + i];
        }
    }

    // Close the file
    fout.close();

    std::cout << "CSV file has been written successfully." << std::endl;
}

template <typename T>
void read_csv(const std::string &filename, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs, T *Q_R, T *q_r, T *A_B, T *d)
{
    // Open the CSV file
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read the CSV line
    std::string line;
    if (!getline(fin, line))
    {
        std::cerr << "Error reading from file: " << filename << std::endl;
        fin.close();
        return;
    }

    // Parse the CSV line
    std::istringstream ss(line);
    std::string token;

    // Read nhorizon, nstates, and ninputs
    if (getline(ss, token, ','))
        nhorizon = std::stoi(token);
    if (getline(ss, token, ','))
        nstates = std::stoi(token);
    if (getline(ss, token, ','))
        ninputs = std::stoi(token);

    // Check dimensions match
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Read Q_R
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < cost_step; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading Q_R from file: " << filename << std::endl;
                return;
            }
            Q_R[timestep * cost_step + i] = std::stod(token);
        }
    }
    // read last Q
    for (int i = 0; i < states_sq; ++i)
    {
        if (!getline(ss, token, ','))
        {
            std::cerr << "Error reading Q_R from file: " << filename << std::endl;
            return;
        }
        Q_R[(nhorizon - 2) * cost_step + i] = std::stod(token);
    }

    // Read q_r
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < nstates + ninputs; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading q_r from file: " << filename << std::endl;
                return;
            }
            q_r[timestep * (nstates + ninputs) + i] = std::stod(token);
        }
    }
    // read last q
    for (int i = 0; i < nstates; ++i)
    {
        if (!getline(ss, token, ','))
        {
            std::cerr << "Error reading q_r from file: " << filename << std::endl;
            return;
        }
        q_r[(nhorizon - 2) * (nstates + ninputs) + i] = std::stod(token);
    }

    // Read A_B
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < dyn_step; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading A_B from file: " << filename << std::endl;
                return;
            }
            A_B[timestep * dyn_step + i] = std::stod(token);
        }
    }

    // Read d
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {
        for (int i = 0; i < nstates; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading d from file: " << filename << std::endl;
                return;
            }
            d[timestep * nstates + i] = std::stod(token);
        }
    }

    fin.close();

    std::cout << "CSV file has been read successfully." << std::endl;
}

template <typename T>
void read_csv(const std::string &filename, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs, T *Q_R, T *q_r, T *A_B, T *d, T *soln)
{
    // Open the CSV file
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read the CSV line
    std::string line;
    if (!getline(fin, line))
    {
        std::cerr << "Error reading from file: " << filename << std::endl;
        fin.close();
        return;
    }

    // Parse the CSV line
    std::istringstream ss(line);
    std::string token;

    // Read nhorizon, nstates, and ninputs
    if (getline(ss, token, ','))
        nhorizon = std::stoi(token);
    if (getline(ss, token, ','))
        nstates = std::stoi(token);
    if (getline(ss, token, ','))
        ninputs = std::stoi(token);

    // Check dimensions match
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Read Q_R
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < cost_step; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading Q_R from file: " << filename << std::endl;
                return;
            }
            Q_R[timestep * cost_step + i] = std::stod(token);
        }
    }
    // read last Q
    for (int i = 0; i < states_sq; ++i)
    {
        if (!getline(ss, token, ','))
        {
            std::cerr << "Error reading Q_R from file: " << filename << std::endl;
            return;
        }
        Q_R[(nhorizon - 2) * cost_step + i] = std::stod(token);
    }

    // Read q_r
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < nstates + ninputs; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading q_r from file: " << filename << std::endl;
                return;
            }
            q_r[timestep * (nstates + ninputs) + i] = std::stod(token);
        }
    }
    // read last q
    for (int i = 0; i < nstates; ++i)
    {
        if (!getline(ss, token, ','))
        {
            std::cerr << "Error reading q_r from file: " << filename << std::endl;
            return;
        }
        q_r[(nhorizon - 2) * (nstates + ninputs) + i] = std::stod(token);
    }

    // Read A_B
    for (int timestep = 0; timestep < nhorizon - 1; ++timestep)
    {
        for (int i = 0; i < dyn_step; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading A_B from file: " << filename << std::endl;
                return;
            }
            A_B[timestep * dyn_step + i] = std::stod(token);
        }
    }

    // Read d
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {
        for (int i = 0; i < nstates; ++i)
        {
            if (!getline(ss, token, ','))
            {
                std::cerr << "Error reading d from file: " << filename << std::endl;
                return;
            }
            d[timestep * nstates + i] = std::stod(token);
        }
    }
    // Read soln
    int soln_size = nstates * nhorizon + ((nstates + ninputs) * nhorizon) - ninputs;
    for (int timestep = 0; timestep < soln_size; ++timestep)
    {
        if (!getline(ss, token, ','))
        {
            std::cerr << "Error reading soln from file: " << filename << std::endl;
            return;
        }
        soln[timestep] = std::stod(token);
    }

    // Close the file
    fin.close();
    std::cout << "CSV file has been read successfully." << std::endl;
}