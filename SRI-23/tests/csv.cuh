// An experiment file to remind myself how I/O works in C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
using namespace std;

/* Creates csv from arrays
 Double check row/column order
 */
template <typename T>
void create_csv(uint32_t nhorizon, uint32_t nstates, uint32_t ninputs,
                T *Q_R, T *q_r, T *A_B, T *d)
{
    // file pointer
    fstream fout;

    fout.open("csv_example.csv", ios::out);
    // Check if the file is successfully opened
    if (!fout.is_open())
    {
        cerr << "Error opening file for writing" << endl;
        return;
    }
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    //write general info of KKT
    fout<<nhorizon<<", "
    <<nstates<<", "
    <<ninputs<<", "
    <<"\n";
    // write Q_R
    for (int timestep = 0; timestep < nhorizon; timestep++)
    {
        for (int i = 0; i < cost_step; i++)
        {
            // insert the date
            fout << *(Q_R + timestep * cost_step + i) << ", ";
        }
        fout << "\n";
    }

    // Write q-r
    for (int timestep = 0; timestep < nhorizon; timestep++)
    {
        for (int i = 0; i < nstates + ninputs; i++)
        {
            // insert the date
            fout << *(q_r + timestep * (nstates + ninputs) + i) << ", ";
        }
        fout << "\n";
    }

    // write A_B
    for (int timestep = 0; timestep < nhorizon; timestep++)
    {
        for (int i = 0; i < dyn_step; i++)
        {
            fout << *(A_B + timestep * dyn_step + i) << ", ";
        }
        fout << "\n";
    }

    // Write d
    for (int timestep = 0; timestep < nhorizon; timestep++)
    {
        for (int i = 0; i < nstates; i++)
        {
            fout << *(d + timestep * (nstates) + i) << ", ";
        }
        fout << "\n";
    }

    // Close the file
    fout.close();
    cout << "CSV file has been written successfully." << endl;
}


//a function to read_csv
template <typename T>
void read_csv(const string &filename, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs, T *Q_R, T *q_r,
              T *A_B, T *d)
{
    // Open the CSV file
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Read the CSV file and populate the Q_R array
    // Read the matrix details (pointer, nstates, ninputs)
    std::string nhorizon_str, nstates_str, ninputs_str;
    getline(fin, nhorizon_str, ',');
    getline(fin, nstates_str, ',');
    getline(fin, ninputs_str, '\n');
    // Convert string values to appropriate types
    uint32_t nhorizon_read = std::stoi(nhorizon_str);
    uint32_t nstates_read = std::stoi(nstates_str);
    uint32_t ninputs_read = std::stoi(ninputs_str);

    // check that dimensions match
    if (nstates_read != nstates || ninputs_read != ninputs)
    {
        std::cerr << "Mismatched dimensions in the CSV file." << std::endl;
        return;
    }

    // read Q_R
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {

        // Read the matrix values
        for (int i = 0; i < cost_step; ++i)
        {
            std::string value_str;
            getline(fin, value_str, ',');

            // Convert string value to the appropriate type and store it in Q_R
            Q_R[timestep * cost_step + i] = std::stod(value_str);
        }
        // Move to the next line
        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    // read q_r
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {

        // Read the matrix values
        for (int i = 0; i < nstates + ninputs; ++i)
        {
            std::string value_str;
            getline(fin, value_str, ',');

            // Convert string value to the appropriate type and store it in Q_R
            q_r[timestep * (nstates + ninputs) + i] = std::stod(value_str);
        }
        // Move to the next line
        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    // read A_B
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {

        // Read the matrix values
        for (int i = 0; i < dyn_step; ++i)
        {
            std::string value_str;
            getline(fin, value_str, ',');

            // Convert string value to the appropriate type and store it in Q_R
            A_B[timestep * dyn_step + i] = std::stod(value_str);
        }
        // Move to the next line
        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    //read d
    for (int timestep = 0; timestep < nhorizon; ++timestep)
    {

        // Read the matrix values
        for (int i = 0; i < nstates ; ++i)
        {
            std::string value_str;
            getline(fin, value_str, ',');

            // Convert string value to the appropriate type and store it in Q_R
            d[timestep * (nstates) + i] = std::stod(value_str);
        }
        // Move to the next line
        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Close the file
    fin.close();

    std::cout << "CSV file has been read successfully." << std::endl;
}
