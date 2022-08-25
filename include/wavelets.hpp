/*
coefficients are obtained from the GSL package (gsl/wavelet/daubechies.c)
*/

#ifndef __WAVELETS_HPP__
#define __WAVELETS_HPP__

#include "type_declaration.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

static const ScalarType h_2[2] = {M_SQRT1_2, M_SQRT1_2};

static const ScalarType g_2[2] = {M_SQRT1_2, -M_SQRT1_2};

static const ScalarType h_4[4] = {
    0.48296291314453414337487159986, 0.83651630373780790557529378092, 0.22414386804201338102597276224,
    -0.12940952255126038117444941881};

static const ScalarType g_4[4] = {
    -0.12940952255126038117444941881, -0.22414386804201338102597276224, 0.83651630373780790557529378092,
    -0.48296291314453414337487159986};

static const ScalarType h_6[6] = {0.33267055295008261599851158914,  0.80689150931109257649449360409,
                                  0.45987750211849157009515194215,  -0.13501102001025458869638990670,
                                  -0.08544127388202666169281916918, 0.03522629188570953660274066472};

static const ScalarType g_6[6] = {0.03522629188570953660274066472,  0.08544127388202666169281916918,
                                  -0.13501102001025458869638990670, -0.45987750211849157009515194215,
                                  0.80689150931109257649449360409,  -0.33267055295008261599851158914};

static const ScalarType h_8[8] = {0.23037781330889650086329118304,  0.71484657055291564708992195527,
                                  0.63088076792985890788171633830,  -0.02798376941685985421141374718,
                                  -0.18703481171909308407957067279, 0.03084138183556076362721936253,
                                  0.03288301166688519973540751355,  -0.01059740178506903210488320852};

static const ScalarType g_8[8] = {-0.01059740178506903210488320852, -0.03288301166688519973540751355,
                                  0.03084138183556076362721936253,  0.18703481171909308407957067279,
                                  -0.02798376941685985421141374718, -0.63088076792985890788171633830,
                                  0.71484657055291564708992195527,  -0.23037781330889650086329118304};

static const ScalarType h_10[10] = {0.16010239797419291448072374802,  0.60382926979718967054011930653,
                                    0.72430852843777292772807124410,  0.13842814590132073150539714634,
                                    -0.24229488706638203186257137947, -0.03224486958463837464847975506,
                                    0.07757149384004571352313048939,  -0.00624149021279827427419051911,
                                    -0.01258075199908199946850973993, 0.00333572528547377127799818342};

static const ScalarType g_10[10] = {0.00333572528547377127799818342,  0.01258075199908199946850973993,
                                    -0.00624149021279827427419051911, -0.07757149384004571352313048939,
                                    -0.03224486958463837464847975506, 0.24229488706638203186257137947,
                                    0.13842814590132073150539714634,  -0.72430852843777292772807124410,
                                    0.60382926979718967054011930653,  -0.16010239797419291448072374802};

static const ScalarType h_12[12] = {
    0.11154074335010946362132391724, 0.49462389039845308567720417688,  0.75113390802109535067893449844,
    0.31525035170919762908598965481, -0.22626469396543982007631450066, -0.12976686756726193556228960588,
    0.09750160558732304910234355254, 0.02752286553030572862554083950,  -0.03158203931748602956507908070,
    0.00055384220116149613925191840, 0.00477725751094551063963597525,  -0.00107730108530847956485262161};

static const ScalarType g_12[12] = {
    -0.00107730108530847956485262161, -0.00477725751094551063963597525, 0.00055384220116149613925191840,
    0.03158203931748602956507908070,  0.02752286553030572862554083950,  -0.09750160558732304910234355254,
    -0.12976686756726193556228960588, 0.22626469396543982007631450066,  0.31525035170919762908598965481,
    -0.75113390802109535067893449844, 0.49462389039845308567720417688,  -0.11154074335010946362132391724};

static const ScalarType h_14[14] = {
    0.07785205408500917901996352196,  0.39653931948191730653900039094,  0.72913209084623511991694307034,
    0.46978228740519312247159116097,  -0.14390600392856497540506836221, -0.22403618499387498263814042023,
    0.07130921926683026475087657050,  0.08061260915108307191292248036,  -0.03802993693501441357959206160,
    -0.01657454163066688065410767489, 0.01255099855609984061298988603,  0.00042957797292136652113212912,
    -0.00180164070404749091526826291, 0.00035371379997452024844629584};

static const ScalarType g_14[14] = {
    0.00035371379997452024844629584,  0.00180164070404749091526826291,  0.00042957797292136652113212912,
    -0.01255099855609984061298988603, -0.01657454163066688065410767489, 0.03802993693501441357959206160,
    0.08061260915108307191292248036,  -0.07130921926683026475087657050, -0.22403618499387498263814042023,
    0.14390600392856497540506836221,  0.46978228740519312247159116097,  -0.72913209084623511991694307034,
    0.39653931948191730653900039094,  -0.07785205408500917901996352196};

static const ScalarType h_16[16] = {
    0.05441584224310400995500940520,  0.31287159091429997065916237551,  0.67563073629728980680780076705,
    0.58535468365420671277126552005,  -0.01582910525634930566738054788, -0.28401554296154692651620313237,
    0.00047248457391328277036059001,  0.12874742662047845885702928751,  -0.01736930100180754616961614887,
    -0.04408825393079475150676372324, 0.01398102791739828164872293057,  0.00874609404740577671638274325,
    -0.00487035299345157431042218156, -0.00039174037337694704629808036, 0.00067544940645056936636954757,
    -0.00011747678412476953373062823};

static const ScalarType g_16[16] = {
    -0.00011747678412476953373062823, -0.00067544940645056936636954757, -0.00039174037337694704629808036,
    0.00487035299345157431042218156,  0.00874609404740577671638274325,  -0.01398102791739828164872293057,
    -0.04408825393079475150676372324, 0.01736930100180754616961614887,  0.12874742662047845885702928751,
    -0.00047248457391328277036059001, -0.28401554296154692651620313237, 0.01582910525634930566738054788,
    0.58535468365420671277126552005,  -0.67563073629728980680780076705, 0.31287159091429997065916237551,
    -0.05441584224310400995500940520};

static const ScalarType h_18[18] = {
    0.03807794736387834658869765888,  0.24383467461259035373204158165,  0.60482312369011111190307686743,
    0.65728807805130053807821263905,  0.13319738582500757619095494590,  -0.29327378327917490880640319524,
    -0.09684078322297646051350813354, 0.14854074933810638013507271751,  0.03072568147933337921231740072,
    -0.06763282906132997367564227483, 0.00025094711483145195758718975,  0.02236166212367909720537378270,
    -0.00472320475775139727792570785, -0.00428150368246342983449679500, 0.00184764688305622647661912949,
    0.00023038576352319596720521639,  -0.00025196318894271013697498868, 0.00003934732031627159948068988};

static const ScalarType g_18[18] = {
    0.00003934732031627159948068988,  0.00025196318894271013697498868,  0.00023038576352319596720521639,
    -0.00184764688305622647661912949, -0.00428150368246342983449679500, 0.00472320475775139727792570785,
    0.02236166212367909720537378270,  -0.00025094711483145195758718975, -0.06763282906132997367564227483,
    -0.03072568147933337921231740072, 0.14854074933810638013507271751,  0.09684078322297646051350813354,
    -0.29327378327917490880640319524, -0.13319738582500757619095494590, 0.65728807805130053807821263905,
    -0.60482312369011111190307686743, 0.24383467461259035373204158165,  -0.03807794736387834658869765888};

static const ScalarType h_20[20] = {
    0.02667005790055555358661744877,  0.18817680007769148902089297368,  0.52720118893172558648174482796,
    0.68845903945360356574187178255,  0.28117234366057746074872699845,  -0.24984642432731537941610189792,
    -0.19594627437737704350429925432, 0.12736934033579326008267723320,  0.09305736460357235116035228984,
    -0.07139414716639708714533609308, -0.02945753682187581285828323760, 0.03321267405934100173976365318,
    0.00360655356695616965542329142,  -0.01073317548333057504431811411, 0.00139535174705290116578931845,
    0.00199240529518505611715874224,  -0.00068585669495971162656137098, -0.00011646685512928545095148097,
    0.00009358867032006959133405013,  -0.00001326420289452124481243668};

static const ScalarType g_20[20] = {
    -0.00001326420289452124481243668, -0.00009358867032006959133405013, -0.00011646685512928545095148097,
    0.00068585669495971162656137098,  0.00199240529518505611715874224,  -0.00139535174705290116578931845,
    -0.01073317548333057504431811411, -0.00360655356695616965542329142, 0.03321267405934100173976365318,
    0.02945753682187581285828323760,  -0.07139414716639708714533609308, -0.09305736460357235116035228984,
    0.12736934033579326008267723320,  0.19594627437737704350429925432,  -0.24984642432731537941610189792,
    -0.28117234366057746074872699845, 0.68845903945360356574187178255,  -0.52720118893172558648174482796,
    0.18817680007769148902089297368,  -0.02667005790055555358661744877};

template <int M> class DaubechiesWaveletBasis {
  private:
    DaubechiesWaveletBasis() = delete;
    ~DaubechiesWaveletBasis() = delete;

    constexpr inline static void get_h_g(const ScalarType *&h_ptr, const ScalarType *&g_ptr) {
        static_assert(1 <= M && M <= 10);
        switch (M) {
        case 1: h_ptr = h_2, g_ptr = g_2; break;
        case 2: h_ptr = h_4, g_ptr = g_4; break;
        case 3: h_ptr = h_6, g_ptr = g_6; break;
        case 4: h_ptr = h_8, g_ptr = g_8; break;
        case 5: h_ptr = h_10, g_ptr = g_10; break;
        case 6: h_ptr = h_12, g_ptr = g_12; break;
        case 7: h_ptr = h_14, g_ptr = g_14; break;
        case 8: h_ptr = h_16, g_ptr = g_16; break;
        case 9: h_ptr = h_18, g_ptr = g_18; break;
        case 10: h_ptr = h_20, g_ptr = g_20; break;
        }
    }

  public:
    template <class Derived>
    static void decompose_mat(
        const Eigen::DenseBase<Derived> &mat, SparseMatrix &decomposed, int num_elems = -1, ScalarType eps = 0.0
    ) {
        const ScalarType *h, *g;
        get_h_g(h, g);

        int N = mat.rows();
        using Triplet = Eigen::Triplet<ScalarType>;
        auto compare = [](Triplet &tri1, Triplet &tri2) { return std::abs(tri1.value()) > std::abs(tri2.value()); };
        std::vector<Triplet> elems;

        std::function<void(std::vector<Triplet> &, int, int, ScalarType)> emplace_back;
        if (num_elems > 0) {
            elems.reserve(num_elems + 1);
            emplace_back = [eps, num_elems, compare](std::vector<Triplet> &elems, int i, int j, ScalarType value) {
                if (std::abs(value) > eps) {
                    elems.emplace_back(i, j, value);
                    std::push_heap(elems.begin(), elems.end(), compare);
                    if (elems.size() > num_elems) {
                        std::pop_heap(elems.begin(), elems.end(), compare);
                        elems.pop_back();
                    }
                }
            };
        } else {
            elems.reserve(N * N);
            emplace_back = [eps](std::vector<Triplet> &elems, int i, int j, ScalarType value) {
                if (std::abs(value) > eps) elems.emplace_back(i, j, value);
            };
        }

        MatrixXs s;
        int size = N / 2;
        int offset = 0;
        while (size >= 1) {
            MatrixXs new_s = MatrixXs::Zero(size, size);
            for (int i = 0; i < size; i++) {
                for (int l = 0; l < size; l++) {
                    ScalarType _alpha = 0.0, _beta = 0.0, _gamma = 0.0;
                    for (int k = 0; k < 2 * M; k++) {
                        for (int m = 0; m < 2 * M; m++) {
                            ScalarType s_prev = offset == 0 ? mat((k + 2 * i) % (2 * size), (m + 2 * l) % (2 * size))
                                                            : s((k + 2 * i) % (2 * size), (m + 2 * l) % (2 * size));
                            _alpha += g[k] * g[m] * s_prev;
                            _beta += g[k] * h[m] * s_prev;
                            _gamma += h[k] * g[m] * s_prev;
                            new_s(i, l) += h[k] * h[m] * s_prev;
                        }
                    }
                    emplace_back(elems, offset + i, offset + l, _alpha);
                    emplace_back(elems, offset + i, offset + size + l, _beta);
                    emplace_back(elems, offset + size + i, offset + l, _gamma);
                }
            }

            offset += 2 * size;
            size /= 2;
            s.swap(new_s);
        }
        emplace_back(elems, 2 * N - 3, 2 * N - 3, s(0, 0));

        decomposed.resize(2 * N - 2, 2 * N - 2);
        decomposed.setFromTriplets(elems.begin(), elems.end());
    }

    // Step 2
    static void decompose_vec(const VectorXs &vec, VectorXs &decomposed) {
        const ScalarType *h, *g;
        get_h_g(h, g);

        int N = vec.rows();

        decomposed = VectorXs::Zero(2 * N - 2);
        const ScalarType *s = vec.data();
        int size = N / 2;
        int offset = 0;
        while (size >= 1) {
            ScalarType *d = decomposed.data() + offset;
            ScalarType *new_s = decomposed.data() + offset + size;

            for (int _s = 0; _s < 2 * size; _s++) {
                ScalarType s_prev = s[_s];
                if (s_prev == 0.0) continue;
#pragma unroll
                for (int n = _s % 2; n < 2 * M; n += 2) {
                    int k = ((_s - n) / 2 + size) % size;
                    new_s[k] += h[n] * s_prev;
                    d[k] += g[n] * s_prev;
                }
            }

            offset += 2 * size;
            size /= 2;
            s = new_s;
        }
    }

    // Step 4
    static void reconstruct(const VectorXs &vec, VectorXs &res) {
        const ScalarType *h, *g;
        get_h_g(h, g);

        int N = (vec.rows() + 2) / 2;
        int offset = -2;
        int size = 1;
        res = VectorXs::Zero(N);
        VectorXs s(N / 2);
        while (size <= N / 2) {
            auto d = vec.segment(2 * N - 2 + offset, size);
            s.head(size) = vec.segment(2 * N - 2 + offset + size, size) + res.head(size);

            res.head(2 * size).setZero();
            // #pragma omp parallel for if(size>32)
            for (int n = 0; n < size; n++) {
#pragma unroll
                for (int k = 0; k < M; k++) {
                    int index = (size + n - k % size) % size;
                    res(2 * n) += h[2 * k] * s[index] + g[2 * k] * d[index];
                    res(2 * n + 1) += h[2 * k + 1] * s[index] + g[2 * k + 1] * d[index];
                }
            }

            offset -= 4 * size;
            size *= 2;
        }
    }

    template <typename Derived>
    static VectorXs
    reconstruct_rows(const SparseMatrix &mat, const VectorXs &vec, const Eigen::DenseBase<Derived> &rows, int N) {
        const ScalarType *h, *g;
        get_h_g(h, g);

        std::unordered_map<int, ScalarType> values;
        auto get_row = [&](int row, int offset) -> ScalarType {
            if (values.count(row + offset) == 0) values[row + offset] = (mat.row(row + offset) * vec).value();
            return values[row + offset];
        };

        std::function<ScalarType(int, int, int)> get_scale = [&](int row, int size, int offset) -> ScalarType {
            if (size == 0) return 0.0;

            ScalarType res = 0.0;
            for (int n = row % 2; n < 2 * M; n += 2) {
                int k = ((row - n) / 2 + size) % size;

                res += h[n] * (get_row(k, offset + size) + get_scale(k, size / 2, offset + 2 * size));
                res += g[n] * get_row(k, offset);
            }

            return res;
        };

        int N_padded = (mat.rows() + 2) / 2;
        VectorXs res = VectorXs::Zero(N);
        for (int i = 0; i < rows.size(); i++) { res[rows[i]] = get_scale(rows[i], N_padded / 2, 0); }
        return res;
    }

    static void update_elem(VectorXs &decomposed_vec, int row, ScalarType diff) {
        const ScalarType *h, *g;
        get_h_g(h, g);

        std::function<void(int, int, int, ScalarType)> update_elems = [&](int row, int size, int offset,
                                                                          ScalarType diff) {
            if (size == 0) return;

            for (int n = row % 2; n < 2 * M; n += 2) {
                int k = ((row - n) / 2 + size) % size;

                decomposed_vec[offset + size + k] += h[n] * diff;
                decomposed_vec[offset + k] += g[n] * diff;
                update_elems(k, size / 2, offset + 2 * size, h[n] * diff);
            }
        };

        int N = (decomposed_vec.rows() + 2) / 2;
        update_elems(row, N / 2, 0, diff);
    }
};

template <size_t M> class CompressedVector;

template <size_t M> class CompressedMatrix;

template <size_t M = 1> class CompressedMatrixVectorProduct {
  public:
    CompressedMatrixVectorProduct(int N = 0) : N(N) {
        data = VectorXs::Zero(2 * std::round(std::pow(2, std::ceil(std::log2(std::max(N, 2))))) - 2);
    }
    CompressedMatrixVectorProduct(const VectorXs &data, int N) : data(data), N(N) {}
    CompressedMatrixVectorProduct(const CompressedMatrix<M> &mat, const CompressedVector<M> &vec) {
        this->data = mat.data * vec.data;
        this->N = mat.N;
    }

    template <typename Derived>
    static VectorXs
    rows(const CompressedMatrix<M> &mat, const CompressedVector<M> &vec, const Eigen::DenseBase<Derived> &rows) {
        return DaubechiesWaveletBasis<M>::reconstruct_rows(mat.data, vec.data, rows, mat.N);
    }
    void operator=(const CompressedMatrixVectorProduct &other) {
        N = other.N;
        data = other.data;
    }
    CompressedMatrixVectorProduct operator+(const CompressedMatrixVectorProduct &other) const {
        return CompressedMatrixVectorProduct(this->data + other.data, N);
    }

    CompressedMatrixVectorProduct &operator+=(const CompressedMatrixVectorProduct &other) {
        this->data += other.data;
        return *this;
    }

    operator VectorXs() const {
        VectorXs res;
        DaubechiesWaveletBasis<M>::reconstruct(data, res);
        return res.head(N);
    }

  public:
    VectorXs data;
    int N;
};

template <size_t M = 1> class CompressedVector {
  public:
    CompressedVector(){};

    template <typename Derived> CompressedVector(const Eigen::DenseBase<Derived> &vec) {
        int N = vec.size();
        int N_padded = std::round(std::pow(2, std::ceil(std::log2(N))));

        VectorXs vec_padded = VectorXs::Zero(N_padded);
        vec_padded.head(N) = vec;
        DaubechiesWaveletBasis<M>::decompose_vec(vec_padded, data);
    };

    void operator=(const CompressedVector &vec) { data = vec.data; };

    void update_elem(int row, ScalarType diff) { DaubechiesWaveletBasis<M>::update_elem(data, row, diff); };

  public:
    VectorXs data;
};

template <size_t M = 1> class CompressedMatrix {
  public:
    CompressedMatrix(int N = 0) : N(N) {
        int N_padded = std::round(std::pow(2, std::ceil(std::log2(N))));
        if (N >= 1) data.resize(2 * N_padded - 2, 2 * N_padded - 2);
    };

    CompressedMatrix(const MatrixXs &mat, int num_elems = -1, ScalarType eps = 0.0) {
        N = mat.rows();
        int N_padded = std::round(std::pow(2, std::ceil(std::log2(N))));

        auto mat_padded = MatrixXs::NullaryExpr(N_padded, N_padded, [&mat](Eigen::Index i, Eigen::Index j) {
            return i < mat.rows() && j < mat.cols() ? mat(i, j) : 0.0;
        });

        DaubechiesWaveletBasis<M>::decompose_mat(mat_padded, data, num_elems, eps);
    };

    // compression ratio here is the ratio of coefficients to be kept.
    CompressedMatrix(const MatrixXs &mat, const ScalarType compression_ratio)
        : CompressedMatrix(mat, (int)(mat.size() * compression_ratio), 0.0){};

    void operator=(const CompressedMatrix &mat) {
        N = mat.N;
        data = mat.data;
    };

    CompressedMatrixVectorProduct<M> operator*(const CompressedVector<M> &vec) const {
        return CompressedMatrixVectorProduct<M>(*this, vec);
    };

  public:
    SparseMatrix data;
    int N;
};

template <size_t M> bool load_compressed_matrix(CompressedMatrix<M> &cpmat, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "load_matrix failed: " << filename << std::endl;
        return false;
    }

    in.read((char *)&cpmat.N, sizeof(int));

    using SparseMatrixType = typeof(cpmat.data);
    using _Scalar = typename SparseMatrixType::Scalar;
    using _StorageIndex = typename SparseMatrixType::StorageIndex;

    Eigen::Index rows, cols, nonZeros, outerSize, innerSize;
    in.read((char *)&rows, sizeof(Eigen::Index));
    in.read((char *)&cols, sizeof(Eigen::Index));
    in.read((char *)&nonZeros, sizeof(Eigen::Index));
    in.read((char *)&outerSize, sizeof(Eigen::Index));
    in.read((char *)&innerSize, sizeof(Eigen::Index));

    cpmat.data.resize(rows, cols);
    cpmat.data.makeCompressed();
    cpmat.data.resizeNonZeros(nonZeros);

    in.read((char *)cpmat.data.valuePtr(), sizeof(_Scalar) * nonZeros);
    in.read((char *)cpmat.data.outerIndexPtr(), sizeof(_StorageIndex) * outerSize);
    in.read((char *)cpmat.data.innerIndexPtr(), sizeof(_StorageIndex) * nonZeros);

    cpmat.data.finalize();
    in.close();
    return true;
}

template <size_t M> bool save_compressed_matrix(CompressedMatrix<M> &cpmat, const std::string &filename) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        std::cerr << "save_matrix failed: " << filename << std::endl;
        return false;
    }

    out.write((const char *)&cpmat.N, sizeof(int));
    cpmat.data.makeCompressed();

    using SparseMatrixType = typeof(cpmat.data);
    using _Scalar = typename SparseMatrixType::Scalar;
    using _StorageIndex = typename SparseMatrixType::StorageIndex;

    Eigen::Index rows = cpmat.data.rows();
    Eigen::Index cols = cpmat.data.cols();
    Eigen::Index nonZeros = cpmat.data.nonZeros();
    Eigen::Index outerSize = cpmat.data.outerSize();
    Eigen::Index innerSize = cpmat.data.innerSize();

    out.write((const char *)&rows, sizeof(Eigen::Index));
    out.write((const char *)&cols, sizeof(Eigen::Index));
    out.write((const char *)&nonZeros, sizeof(Eigen::Index));
    out.write((const char *)&outerSize, sizeof(Eigen::Index));
    out.write((const char *)&innerSize, sizeof(Eigen::Index));

    out.write((const char *)cpmat.data.valuePtr(), sizeof(_Scalar) * nonZeros);
    out.write((const char *)cpmat.data.outerIndexPtr(), sizeof(_StorageIndex) * outerSize);
    out.write((const char *)cpmat.data.innerIndexPtr(), sizeof(_StorageIndex) * nonZeros);

    out.close();
    return true;
}

#endif //!__WAVELETS_HPP__