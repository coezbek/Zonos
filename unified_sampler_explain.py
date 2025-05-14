from zonos.sampling import print_unified_sampler_explanation

linear = 0.72
quad = (1/3)-(linear*4/15)
conf = -0.5*quad

print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

# What effect does making linear smaller/bigger have?
print("Bigger linear: 0.72 -> 0.9")
linear = 0.9
quad = (1/3)-(linear*4/15)
conf = -0.5*quad
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

print("Smaller linear: 0.72 -> 0.55")
linear = 0.55
quad = (1/3)-(linear*4/15)
conf = -0.5*quad
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

# What effect does making quad smaller/bigger have?
print("Bigger quad: 1/3 - linear*3/15, instead of 4/15")
linear = 0.72
quad = (1/3)-(linear*3/15)
conf = -0.5*quad
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

print("Smaller quad: 1/3 - linear*5/15, instead of 4/15")
linear = 0.72
quad = (1/3)-(linear*5/15)
conf = -0.5*quad
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

# What effect does making conf smaller/bigger have?
print("Smaller conf: -0.7 * quad (instead of -0.5 * quad)")
linear = 0.72
quad = (1/3)-(linear*4/15)
conf = -0.7*quad
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

print("Bigger conf: -0.3 * quad (instead of -0.5 * quad)")
linear = 0.72
quad = (1/3)-(linear*4/15)
conf = -0.3*quad # 
print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)

# This was the original config in the code: 
print("Original config: linear=0.6, conf=0.4, quad=0.0 in the python code")
print_unified_sampler_explanation(linear=0.6, conf=0.4, quad=0.0)

print("My config after laying a lot: linear=0.6, conf=0.4, quad=0.0 in the python code")
print_unified_sampler_explanation(linear=0.8, conf=0.2, quad=0.0)