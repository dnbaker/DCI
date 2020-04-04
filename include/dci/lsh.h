#ifndef DCI_LSH_H__
#define DCI_LSH_H__
#include "blaze/Math.h"
#include "clhash_ho.h"
#include "macros.h"

namespace dci {
namespace hash {

struct mclhasher {
    const void *random_data_;
    mclhasher(uint64_t seed1=137, uint64_t seed2=777): random_data_(get_random_key_for_clhash(seed1, seed2)) {}
    mclhasher(const mclhasher &o): random_data_(copy_random_data(o)) {} // copy data
    mclhasher(mclhasher &&o): random_data_(o.random_data_) {
        o.random_data_ = nullptr; // move
    }
    static void *copy_random_data(const mclhasher &o) {
        void *ret;
        if(posix_memalign(&ret, sizeof(__m128i), RANDOM_BYTES_NEEDED_FOR_CLHASH)) throw std::bad_alloc();
        return std::memcpy(ret, o.random_data_, RANDOM_BYTES_NEEDED_FOR_CLHASH);
    }
    template<typename T>
    uint64_t operator()(const T *data, const size_t len) const {
        return clhash(random_data_, (const char *)data, len * sizeof(T));
    }
    uint64_t operator()(const char *str) const {return operator()(str, std::strlen(str));}
    template<typename T>
    uint64_t operator()(const T &input) const {
        return operator()((const char *)&input, sizeof(T));
    }
    template<typename T>
    uint64_t operator()(const std::vector<T> &input) const {
        return operator()((const char *)input.data(), sizeof(T) * input.size());
    }
    uint64_t operator()(const std::string &str) const {
        return operator()(str.data(), str.size());
    }
    ~mclhasher() {
        std::free((void *)random_data_);
    }
};
template<typename F, typename V> ATTR_CONST INLINE auto cmp_zero(V v);
#if _FEATURE_AVX512F
template<> ATTR_CONST INLINE auto
cmp_zero<float> (__m512 v) {
    return _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
}
template<> ATTR_CONST INLINE auto
cmp_zero<float> (__m512d v) {
    return _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_GT_OQ);
}
#elif __AVX__
template<> 
ATTR_CONST INLINE
auto cmp_zero<float, __m256> (__m256 v) {
    return _mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
}
template<>
ATTR_CONST INLINE
auto cmp_zero<double, __m256d> (__m256d v) {
    return _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_GT_OQ));
}
#else
#pragma message("not vectorizing signed projection hashing")
#endif

template<typename FType, bool SO>
struct empty {
    template<typename...Args> empty(Args &&...args) {}
};

template<typename FType, size_t VSZ>
struct F2VType;
#if __AVX__

template<> struct F2VType<float, 32> {
    using type = __m256;
    static type load(const float *a) {
        return _mm256_loadu_ps(a);
    }
};
template<> struct F2VType<double, 32> {
    using type = __m256d;
    static type load(const double *a) {
        return _mm256_loadu_pd(a);
    }
};
#endif
#if __AVX512F__
template<> struct F2VType<float, 64> {
    using type = __m512;
    static type load(const float *a) {
        return _mm512_loadu_ps(a);
    }
};
template<> struct F2VType<double, 64> {
    using type = __m512d;
    static type load(const double *a) {
        return _mm512_loadu_pd(a);
    }
};
#endif

#if __AVX512F__
    template<typename FType>
    static constexpr int f2b(__m512d v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    template<typename FType>
    static constexpr int f2b(__m512 v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
#endif
#if __AVX__
    template<typename FType>
    static constexpr int f2b(__m256d v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    template<typename FType>
    static constexpr int f2b(__m256 v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
#endif

template<typename FType, bool SO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>, typename...DistArgs>
blaze::DynamicMatrix<FType, SO>
generate_randproj_matrix(size_t nr, size_t ncol,
                         bool orthonormalize=true, uint64_t seed=0,
                         DistArgs &&...args)
{
    using matrix_type = blaze::DynamicMatrix<FType, SO>;
    matrix_type ret(nr, ncol);
    seed = ((seed ^ nr) * ncol) * seed;
    if(orthonormalize) {
        try {
            matrix_type r, q;
            if(ret.rows() >= ret.columns()) {
                // Randomize
                OMP_PRAGMA("omp parallel for")
                for(size_t i = 0; i < ret.rows(); ++i) {
                    blaze::DefaultRNG gen(seed + i * seed + i);
                    DistributionType dist(std::forward<DistArgs>(args)...);
                    for(auto &v: row(ret, i))
                        v = dist(gen);
                }
                // QR
                blaze::qr(ret, q, r);
                assert(ret.columns() == q.columns());
                assert(ret.rows() == q.rows());
                swap(ret, q);
            } else {
                // Generate random matrix for (C, C) and then just take the first R rows
                const auto mc = ret.columns();
                matrix_type tmp(mc, mc);
                OMP_PRAGMA("omp parallel for")
                for(size_t i = 0; i < tmp.rows(); ++i) {
                    blaze::DefaultRNG gen(seed + i * seed + i);
                    DistributionType dist(std::forward<DistArgs>(args)...);
                    for(auto &v: row(tmp, i))
                        v = dist(gen);
                }
                blaze::qr(tmp, q, r);
                ret = submatrix(q, 0, 0, ret.rows(), ret.columns());
            }
            OMP_PRAGMA("omp parallel for")
            for(size_t i = 0; i < ret.rows(); ++i)
                blaze::normalize(row(ret, i));
        } catch(const std::exception &ex) { // Orthonormalize
            std::fprintf(stderr, "failure in orthonormalization: %s\n", ex.what());
            throw;
        }
    } else {
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < nr; ++i) {
            blaze::DefaultRNG gen(seed + i);
            std::normal_distribution dist;
            for(auto &v: row(ret, i))
                v = dist(gen);
            normalize(row(ret, i));
        }
    }
    return ret;
}



template<typename FType=float, template<typename, bool> class Container=::blaze::DynamicVector, bool SO=blaze::rowMajor>
struct LSHasher {
    using CType = Container<FType, SO>;
    CType container_;
    template<typename... CArgs>
    LSHasher(CArgs &&...args): container_(std::forward<CArgs>(args)...) {}
    template<typename T>
    auto dot(const T &ov) const {
        return blaze::dot(container_, ov);
    }
    // TODO: Store full matrix to get hashes
    // TODO: Use structured matrices to speed up calculation (FFHT, then downsample to bins)
};



template<typename FType, bool OSO>
static INLINE uint64_t cmp2hash(const blaze::DenseVector<FType, OSO> &c, size_t n=0) {
    const auto &cref = ~c;
    assert(n <= 64);
    uint64_t ret = 0;
    if(n == 0) {
        n = n;
    }
#if __AVX512F__
    static constexpr size_t COUNT = sizeof(__m512d) / sizeof(FType);
#elif __AVX__
    static constexpr size_t COUNT = sizeof(__m256d) / sizeof(FType);
#else
    static constexpr size_t COUNT = 0;
#endif
    size_t i = 0;
#if __AVX512F__ || defined(__AVX__)
    CONST_IF(COUNT) {
static constexpr size_t VSIZE =
#if __AVX512F__
        64
#elif __AVX__
        32
#else
#error("Can't get here")
#endif
        ;
        using LV = F2VType<FType, VSIZE>;
        for(; i < n / COUNT;ret = (ret << COUNT) | cmp_zero<FType, typename LV::type>(LV::load((&cref[i++ * COUNT]))));
        i *= COUNT;
    }
#else
    for(;i + 8 <= n; i += 8) {
        ret = (ret << 8) |
              ((cref[i] > 0.) << 7) | ((cref[i + 1] > 0.) << 6) |
              ((cref[i + 2] > 0.) << 5) | ((cref[i + 3] > 0.) << 4) |
              ((cref[i + 4] > 0.) << 3) | ((cref[i + 5] > 0.) << 2) |
              ((cref[i + 6] > 0.) << 1) | (cref[i + 7] > 0.);
    }
#endif
    for(; i < n; ret = (ret << 1) | (cref[i++] > 0.));
    return ret;
}

template<typename FType=float, bool SO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>>
struct MatrixLSHasher {
    // Hyperplane hasher
    using CType = ::blaze::DynamicMatrix<FType, SO>;
    using this_type       =       MatrixLSHasher<FType, SO>;
    using const_this_type = const MatrixLSHasher<FType, SO>;
    CType container_;
    template<typename...DistArgs>
    MatrixLSHasher(size_t nr, size_t nc, bool orthonormalize=true, uint64_t seed=0,
                   DistArgs &&...args):
        container_(generate_randproj_matrix<FType, SO, DistributionType>(nr, nc, orthonormalize, seed, std::forward<DistArgs>(args)...)) {}

#if 0
    template<bool OSO>
    auto &multiply(const blaze::DynamicVector<FType, OSO> &c, blaze::DynamicVector<FType, SO> &ret) const {
        //std::fprintf(stderr, "size of input: %zu. size of ret: %zu. Matrix sizes: %zu/%zu\n", c.size(), ret.size(), container_.rows(), container_.columns());
        ret = trans(this->container_ * trans(c));
        //std::fprintf(stderr, "multiplied successfully\n");
        return ret;
    }
#endif
    template<typename VT, bool TF>
    decltype(auto) multiply(const blaze::DenseVector<VT, TF> &c) const {
        blaze::DynamicVector<FType, SO> vec;
        if constexpr(TF == SO) {
            vec = this->container_ * ~c;
        } else {
            vec = this->container_ * trans(~c);
        }
        return vec;
    }
#if 0
    auto multiply(const blaze::DynamicVector<FType, SO> &c) const {
        //std::fprintf(stderr, "size of input: %zu. size of vec: %zu. Matrix sizes: %zu/%zu\n", c.size(), container_.rows(), container_.columns());
        blaze::DynamicVector<FType, SO> vec = this->container_ * trans(c);
        return vec;
    }
#endif
    template<typename...Args>
    decltype(auto) project(Args &&...args) const {return multiply(std::forward<Args>(args)...);}
    template<typename VT, bool OSO>
    uint64_t hash(const blaze::DenseVector<VT, OSO> &c) const {
#if VERBOSE_AF
        std::cout << this->container_ << '\n';
#endif
        blaze::DynamicVector<FType, SO> vec = multiply(c);
        return cmp2hash(vec); // This is the SRP hasher (signed random projection)
    }
    template<typename VT, bool OSO>
    uint64_t operator()(const blaze::DenseVector<VT, OSO> &c) const {
        return this->hash(c);
    }
};

template<typename FType=float, bool OSO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>>
struct E2LSHasher {
    MatrixLSHasher<FType, OSO, DistributionType> superhasher_;
    blaze::DynamicVector<FType, OSO> b_;
    const double r_;
    mclhasher clhasher_;
    template<typename...Args>
    E2LSHasher(unsigned d, unsigned k, double r = 1., uint64_t seed=0, Args &&...args):
            superhasher_(/*nhashes = */d, /*dim = */k, false, seed, std::forward<Args>(args)...), r_(r), clhasher_(seed * seed + seed) {
        superhasher_.container_ *= static_cast<FType>(1. / r);
        std::uniform_real_distribution<FType> gen(0, r_);
        std::mt19937_64 mt(seed ^ uint64_t(d * k * r));
        b_ = blaze::generate(d, [&mt,&gen](auto){return gen(mt);});
    }
    E2LSHasher(const E2LSHasher &o) = default;
    E2LSHasher(E2LSHasher &&o) = default;
    template<typename...Args>
    decltype(auto) project(Args &&...args) const {
        return floor(superhasher_.project(std::forward<Args>(args)...) + b_);
    }
    template<typename...Args>
    uint64_t hash(Args &&...args) const {
        auto proj = evaluate(this->project(std::forward<Args>(args)...));
        return clhasher_(&b_[0], b_.size() * sizeof(FType));
    }
    template<typename...Args>
    uint64_t operator()(Args &&...args) const {
        return hash(std::forward<Args>(args)...);
    }
};

template<typename FT=float>
struct ThresholdedCauchyDistribution {
    std::cauchy_distribution<FT> cd_;
    FT absmax_;
    template<typename...Args> ThresholdedCauchyDistribution(FT absmax, Args &&...args): cd_(std::forward<Args>(args)...), absmax_(std::abs(absmax)) {
    }
    FT operator()() {
        return std::clamp(cd_(), -absmax_, absmax_);
    }
};

template<typename FType=float, bool OSO=blaze::rowMajor>
struct L1E2LSHasher: public E2LSHasher<FType, OSO, ThresholdedCauchyDistribution<double>> {
    using super = E2LSHasher<FType, OSO, ThresholdedCauchyDistribution<double>>;
    L1E2LSHasher(unsigned d, unsigned k, double r = 1., uint64_t seed=0, FType amax=1000.): 
        super(d, k, r, seed, amax) {}
};



} // namespace hash
using hash::L1E2LSHasher;
using hash::E2LSHasher;
using hash::MatrixLSHasher;

} // namespace dci

#endif
