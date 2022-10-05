#include "./mpi_context.h"

namespace dgs {
namespace mpi {

int global_data = 10;

void Initialize() {
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }
  if (global_comm == MPI_COMM_NULL) {
    MPI_Comm_dup(MPI_COMM_WORLD, &global_comm);
    MPI_Comm_size(global_comm, &global_comm_size);
    MPI_Comm_rank(global_comm, &local_rank);
  }
}

void Finalize() {
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
  if (global_comm != MPI_COMM_NULL) {
    global_comm = MPI_COMM_NULL;
  }
}

}  // namespace mpi
}  // namespace dgs