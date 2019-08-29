import math

def is_prime(n):
    primes = [False]*2  + (n-1)*[True]
    print(primes)

    for x in range(2, int(math.sqrt(n) + 1)):
        if primes[x]:
            primes[2*x::x] = [False] * (n // x - 1)
    
    return primes[n]

print(is_prime(31))

class Primer( object ):
    