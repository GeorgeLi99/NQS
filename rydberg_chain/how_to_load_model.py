# ... ...
# VQS
import netket as nk
import flax
import flax.serialization

# vs = nk.vqs.MCState(sampler=sampler, model=model,
#     model = model_symm, # Symmetry projection 
#    n_discard_per_chain=32, chunk_size=1024*8, n_samples= N_samples,  )
# print("Variational state:",vs)
# print("Number of parameters:", vs.n_parameters)

######################################################################
# Add few lines after the definition of vs to load the parameters from a file.
######################################################################

# Loaded parameters (path 与 rydberg_nqs_starter 一致，通常在 train/complex128/ 或 train/complex64/)
# key_model = f"model_name"
# load_name = f"{key_model}.mpack"  # 或 os.path.join("train", "complex128", f"{key_model}.mpack")
# with open(load_name, 'rb') as file:
#     vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
# print("Loading parameters ... Number of parameters:", vs.n_parameters)

######################################################################
# ... ...