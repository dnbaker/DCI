

CXXFLAGS+=-O3 -fopenmp -I. -Iinclude -Iblaze -IxxHash -march=native
LDFLAGS+=-llapack

dcitest: src/dcitest.cpp $(wildcard include/dci/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
