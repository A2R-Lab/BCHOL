#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void scal(std::uint32_t n, 
          T const, 
          T *x, 
          cgrps::thread_group g)
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        x[ind] = const;
    }
}
