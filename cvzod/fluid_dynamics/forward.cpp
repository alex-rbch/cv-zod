#include <algorithm>
#include <cmath>
#include <vector>

struct Fluid {
    int n, size, steps, pressure_steps;
    float dh;
    std::vector<float> initial, target, vx, vy, density, wx, wy, nextx, nexty, nextd, pressure, nextp, div;
    Fluid(int width, int horizon, int iterations, const float* rho, const float* goal)
        : n(width), size(n*n), steps(horizon), pressure_steps(iterations), dh(1.0f/n),
          initial(rho,rho+size), target(goal,goal+size), vx(size), vy(size), density(size),
          wx(size), wy(size), nextx(size), nexty(size), nextd(size), pressure(size), nextp(size), div(size) {}

    int wrap(int x) const { int r=x%n; return r<0?r+n:r; }
    void advect(const std::vector<float>& x, const std::vector<float>& y,
                const std::vector<float>& field, std::vector<float>& result) {
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
            int ij=i*n+j;
            float cx=float(i)-x[ij], cy=float(j)-y[ij];
            int left=int(std::floor(cx)), bottom=int(std::floor(cy));
            float s1=cx-float(left), s0=1.0f-s1, t1=cy-float(bottom), t0=1.0f-t1;
            int i0=wrap(left)*n, i1=wrap(left+1)*n, j0=wrap(bottom), j1=wrap(bottom+1);
            result[ij]=s0*(t0*field[i0+j0]+t1*field[i0+j1])+s1*(t0*field[i1+j0]+t1*field[i1+j1]);
        }
    }
    float evaluate(const float* velocity, float* terminal) {
        std::copy(velocity,velocity+size,vx.begin());
        std::copy(velocity+size,velocity+2*size,vy.begin());
        density=initial;
        std::fill(pressure.begin(),pressure.end(),0.0f);
        for (int step=0;step<steps;++step) {
            advect(vx,vy,vx,wx); advect(vx,vy,vy,wy);
            for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
                int ij=i*n+j, im=wrap(i-1)*n+j, ip=wrap(i+1)*n+j, jm=i*n+wrap(j-1), jp=i*n+wrap(j+1);
                div[ij]=0.5f*(wx[ip]-wx[im]+wy[jp]-wy[jm])/dh;
            }
            for (int k=0;k<pressure_steps;++k) {
                for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
                    int ij=i*n+j, im=wrap(i-1)*n+j, ip=wrap(i+1)*n+j, jm=i*n+wrap(j-1), jp=i*n+wrap(j+1);
                    nextp[ij]=0.25f*(-dh*dh*div[ij]+pressure[im]+pressure[ip]+pressure[jm]+pressure[jp]);
                }
                pressure.swap(nextp);
            }
            for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
                int ij=i*n+j, im=wrap(i-1)*n+j, ip=wrap(i+1)*n+j, jm=i*n+wrap(j-1), jp=i*n+wrap(j+1);
                nextx[ij]=wx[ij]-0.5f*(pressure[ip]-pressure[im])/dh;
                nexty[ij]=wy[ij]-0.5f*(pressure[jp]-pressure[jm])/dh;
            }
            advect(nextx,nexty,density,nextd);
            vx.swap(nextx); vy.swap(nexty); density.swap(nextd);
        }
        float loss=0.0f;
        for (int i=0;i<size;++i) {
            float difference=density[i]-target[i];
            loss += difference*difference/float(size);
        }
        if (terminal) std::copy(density.begin(),density.end(),terminal);
        return loss;
    }
};

extern "C" {
void* fluid_create(int n,int steps,int pressure,const float* initial,const float* target) {
    return new Fluid(n,steps,pressure,initial,target);
}
void fluid_destroy(void* context) { delete static_cast<Fluid*>(context); }
float fluid_evaluate(void* context,const float* velocity,float* terminal) {
    return static_cast<Fluid*>(context)->evaluate(velocity,terminal);
}
void fluid_density(void* context,float* terminal) {
    const auto& density=static_cast<Fluid*>(context)->density;
    std::copy(density.begin(),density.end(),terminal);
}
}
