

CXXFLAGS+=-O3 -fopenmp -I. -Iinclude -Iblaze -IxxHash -march=native
LDFLAGS+=-llapack

dcitest: src/dcitest.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
