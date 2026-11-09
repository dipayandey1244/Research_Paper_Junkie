#!/usr/bin/env python3
"""Illustrative continual-learning toy lab; NOT Google's Sleep Framework."""
import random
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def train(w, data, epochs=35, lr=.035, replay=None):
    rng=random.Random(42)
    for _ in range(epochs):
        rows=list(data)+(list(replay) if replay else [])
        rng.shuffle(rows)
        for x,y in rows:
            err=dot(w,x)-y
            for j in range(len(w)): w[j]-=lr*err*x[j]
    return w
def mse(w, data): return sum((dot(w,x)-y)**2 for x,y in data)/len(data)
def make_task(weights, seed):
    rng=random.Random(seed)
    return [([rng.uniform(-1,1) for _ in weights],None) for _ in range(150)]
def label(rows,w):return [(x,dot(w,x)) for x,_ in rows]
def main():
    old=label(make_task([1,0],1),[1,0])
    new=label(make_task([0,1],2),[0,1])
    backbone=train([0.,0.],old)
    sequential=train(backbone.copy(),new)
    replay=train(backbone.copy(),new,replay=old[:40])
    # Frozen backbone + separate residual memory for new-task routing.
    residual=train([0.,0.],[(x,y-dot(backbone,x)) for x,y in new])
    print('Method                old MSE     new MSE')
    print(f'Naive sequential     {mse(sequential,old):9.4f} {mse(sequential,new):11.4f}')
    print(f'Replay                {mse(replay,old):9.4f} {mse(replay,new):11.4f}')
    print(f'Frozen+task memory   {mse(backbone,old):9.4f} {mse([a+b for a,b in zip(backbone,residual)],new):11.4f}')
    print('\nCaveat: task identity is given to the router; no learned MoE, GKD, or RL.')
if __name__=='__main__':main()
