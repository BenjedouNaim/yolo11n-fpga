// Minimal CHW float tensor (batch = 1) and raw .bin I/O.
#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

struct Tensor {
    int c = 0, h = 0, w = 0;
    std::vector<float> d;

    Tensor() = default;
    Tensor(int c_, int h_, int w_) : c(c_), h(h_), w(w_), d(size_t(c_) * h_ * w_, 0.f) {}

    size_t size() const { return d.size(); }
    float& at(int ci, int y, int x) { return d[(size_t(ci) * h + y) * w + x]; }
    float at(int ci, int y, int x) const { return d[(size_t(ci) * h + y) * w + x]; }
    float* ch(int ci) { return d.data() + size_t(ci) * h * w; }
    const float* ch(int ci) const { return d.data() + size_t(ci) * h * w; }
};

inline std::vector<float> read_bin(const std::string& path, size_t expect = 0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    size_t n = size_t(std::ftell(f)) / sizeof(float);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v(n);
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n || (expect && n != expect))
        throw std::runtime_error(path + ": expected " + std::to_string(expect) + " floats, got " + std::to_string(n));
    return v;
}

inline Tensor read_tensor(const std::string& path, int c, int h, int w) {
    Tensor t(c, h, w);
    t.d = read_bin(path, t.size());
    return t;
}
