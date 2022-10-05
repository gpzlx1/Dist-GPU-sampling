#ifndef DGS_MPI_CONTEXT_H_
#define DGS_MPI_CONTEXT_H_

#include <mpi.h>

namespace dgs {
namespace mpi {

extern int global_comm_size;
extern int local_rank;
extern MPI_Comm global_comm;

void Initialize();
void Finalize();
int64_t GetRank();
int64_t GetSize();

}  // namespace mpi
}  // namespace dgs

#endif