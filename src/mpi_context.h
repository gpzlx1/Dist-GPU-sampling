#ifndef DGS_MPI_CONTEXT_H_
#define DGS_MPI_CONTEXT_H_

#include <mpi.h>

namespace dgs {
namespace mpi {

extern int global_data;

void Initialize();
void Finalize();
int GetRank();

MPI_Comm global_comm = MPI_COMM_NULL;
int global_comm_size;
int local_rank;

}  // namespace mpi
}  // namespace dgs

#endif