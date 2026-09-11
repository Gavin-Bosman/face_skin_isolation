# Layer Submodule

PyFAME uses a layer-based architecture, wherein each image or video manipulation is defined as a layer.
Every layer in PyFAME uses the same base structure, taking in a **TimingConfiguration** object - which contains all 
of the temporal information (including but not limited to onset/offset times, rise/fall curves, rise/fall durations),
a **landmark** - the region of application defined by a complete circular path of facial landmark indices, as well as specific additional configuration parameters. 

layers can be stacked or combined to form complex multi-effect manipulations. Every application of one or more layers is performed by calling the top level method **apply_layers()**, which takes in a dataframe of file paths (returned by **make_paths()**) and a list of layers.

::: pyfame.layer._layer.TimingConfiguration

---

::: pyfame.layer._layer.Layer
    options:
      members:
        - compute_weight
        - supports_weight
        - get_layer_parameters
        - apply_layer
      members_order: source

---

::: pyfame.layer._apply_layers
    options:
      members:
        - apply_layers