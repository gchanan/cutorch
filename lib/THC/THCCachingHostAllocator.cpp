#include "THCCachingHostAllocator.h"

#include <cuda_runtime_api.h>
#include <deque>
#include <list>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <utility>


namespace {

struct BlockSize
{
  size_t  size; // allocation size
  void*   ptr;  // host memory pointer

  BlockSize(size_t size, void* ptr=NULL) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize
{
  bool  allocated;    // true if the block is currently allocated
  int   event_count;  // number of outstanding cuda events

  Block(size_t size, void* ptr, bool allocated) :
      BlockSize(size, ptr), allocated(allocated), event_count(0) { }
};

static bool BlockComparator(const BlockSize& a, const BlockSize& b)
{
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

struct HostAllocator
{
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around blocks and available collections
  std::mutex mutex;

  // lock around cuda_events collections
  std::mutex cuda_events_mutex;

  // lock to ensure one thread is processing events -- if we just used
  // cuda_events_mutex, another thread could be inserting and we wouldn't
  // be guaranteed to make forward progress
  std::mutex events_loop_mutex;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;

  // used as a dummy reference since references can't be null
  Block dummy_block_ref = {0, NULL, true};

  // pointers that are ready to be allocated (event_count=0)
  std::set<BlockSize, Comparison> available;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, void*>> cuda_events;

  HostAllocator() : available(BlockComparator) {}

  cudaError_t malloc(void** ptr, size_t size)
  {
    // process outstanding cuda events which may have occurred
    cudaError_t err = processEvents(true);
    if (err != cudaSuccess) {
      return err;
    }

    std::lock_guard<std::mutex> lock(mutex);

    // search for the smallest block which can hold this allocation
    BlockSize search_key(size);
    auto it = available.lower_bound(search_key);
    if (it != available.end()) {
      Block& block = blocks.at(it->ptr);
      THAssert(!block.allocated && block.event_count == 0);
      block.allocated = true;
      *ptr = block.ptr;
      available.erase(it);
      return cudaSuccess;
    }

    // note that cudaHostAlloc may not touch pointer if size is 0
    *ptr = 0;

    // allocate a new block if no cached allocation is found
    err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      return err;
    }

    blocks.insert({*ptr, Block(size, *ptr, true)});
    return cudaSuccess;
  }

  cudaError_t free(void* ptr)
  {
    if (!ptr) {
      return cudaSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex);

    auto it = blocks.find(ptr);
    THAssert(it != blocks.end());

    Block& block = it->second;
    THAssert(block.allocated);

    block.allocated = false;
    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding cuda events
      available.insert(block);
    }
    return cudaSuccess;
  }

  cudaError_t recordEvent(void* ptr, cudaStream_t stream)
  {
    Block &block = dummy_block_ref;
    {
      std::lock_guard<std::mutex> lock(mutex);

      auto it = blocks.find(ptr);
      if (it == blocks.end()) {
        // ignore events for untracked pointers
        return cudaSuccess;
      }

      block = it->second;
      THAssert(block.allocated);
      // ensure the block will not be reused until after we are complete
      block.event_count++;
    }

    // we are done with the block (unless there is an error and we need to
    // rewind), so we can drop the main mutex.
    cudaError_t err = cudaSuccess;
    do {
      // process outstanding cuda events which may have occurred
      err = processEvents(false);
      if (err != cudaSuccess) {
        break;
      }

      // create and record an event in the given stream
      cudaEvent_t event;
      err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (err != cudaSuccess) {
        break;
      }
      err = cudaEventRecord(event, stream);
      if (err != cudaSuccess) {
        break;
      }

      // enter event, need to hold cuda_events_mutex, but not main mutex.
      std::lock_guard<std::mutex> cuda_event_lock(cuda_events_mutex);
      cuda_events.emplace_back(event, ptr);
    } while (false);

    if (err != cudaSuccess) {
      std::lock_guard<std::mutex> lock(mutex);
      block.event_count--;
      if (block.event_count == 0 && !block.allocated) {
        available.insert(block);
      }
    }
    return err;
  }

  cudaError_t processEvents(bool wait)
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.

    // if not waiting, just try to grab events loop mutex once and give up
    // if we can't acquire it
    if (wait) {
      events_loop_mutex.lock();
    } else if (!events_loop_mutex.try_lock()) {
      return cudaSuccess;
    }

    // cudaEventDestroy is slow, do them outside of lock
    std::list<cudaEvent_t> events_to_destroy;
    cudaError_t err_ret = cudaSuccess;
    {
      std::lock_guard<std::mutex> loop_lock(events_loop_mutex, std::adopt_lock);
      std::lock_guard<std::mutex> cuda_event_lock(cuda_events_mutex);
      while (!cuda_events.empty()) {
        auto& e = cuda_events.front();
        cudaEvent_t event = e.first;

        cudaError_t err = cudaEventQuery(event);
        if (err == cudaErrorNotReady) {
          break;
        } else if (err != cudaSuccess) {
          err_ret = err;
          break;
        }

        events_to_destroy.emplace_back(event);
        {
          std::lock_guard<std::mutex> lock(mutex);
          Block& block = blocks.at(e.second);

          block.event_count--;
          if (block.event_count == 0 && !block.allocated) {
            available.insert(block);
          }
        }
        cuda_events.pop_front();
      }
    }

    // now we can detroy events outside of loop
    for (auto event : events_to_destroy) {
      cudaError_t err = cudaEventDestroy(event);

      // continue destroying even though a single event failed,
      // but return first error
      if (err != cudaSuccess && err_ret != cudaSuccess) {
        err_ret = err;
      }
    }
    return err_ret;
  }

  void emptyCache()
  {
    // event loop acquires locks in this order, so we must as well
    std::lock_guard<std::mutex> cuda_event_lock(cuda_events_mutex);
    std::lock_guard<std::mutex> lock(mutex);

    // remove events for freed blocks
    std::deque<std::pair<cudaEvent_t, void*>> new_events;
    for (auto it = cuda_events.begin(); it != cuda_events.end(); ++it) {
      cudaEvent_t event = it->first;
      Block& block = blocks.at(it->second);
      if (!block.allocated) {
        THCudaCheckWarn(cudaEventDestroy(event));
        block.event_count--;
      } else {
        new_events.push_back(*it);
      }
    }
    cuda_events.swap(new_events);

    // clear list of available blocks
    available.clear();

    // free and erase non-allocated blocks
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block& block = it->second;
      if (!block.allocated) {
        THCudaCheckWarn(cudaFreeHost(block.ptr));
        it = blocks.erase(it);
      } else {
        ++it;
      }
    }
  }
};

}  // namespace

static HostAllocator allocator;

static void* THCCachingHostAllocator_malloc(void* ctx, ptrdiff_t size)
{
  THAssert(size >= 0);
  void *ptr;
  THCudaCheck(allocator.malloc(&ptr, size));
  return ptr;
}

static void THCCachingHostAllocator_free(void* ctx, void* ptr)
{
  allocator.free(ptr);
}

cudaError_t THCCachingHostAllocator_recordEvent(void *ptr, cudaStream_t stream)
{
  return allocator.recordEvent(ptr, stream);
}

void THCCachingHostAllocator_emptyCache()
{
  allocator.emptyCache();
}

THAllocator THCCachingHostAllocator = {
  &THCCachingHostAllocator_malloc,
  NULL,
  &THCCachingHostAllocator_free,
};
