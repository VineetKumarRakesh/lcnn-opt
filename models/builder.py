# models/builder.py

import torchvision.models as tv_models
import timm
import torch

def get_model(name: str, pretrained: bool = False, num_classes: int = 1000) -> torch.nn.Module:
    """
    Factory to create a model by name.
    Supported names:
      - "efficientnetv2_s"
      - "convnext_tiny"
      - "mobilenetv3_large_100"
      - "mobilevitv2_xs"
      - "mobilevitv2_s"
      - "tiny_vit_21m_224"
      - "repvgg_a2"
    """

    # List everything containing "mobilevit"
    print(timm.list_models("*mobilevit*"))
    # List everything containing "tiny_vit"
    print(timm.list_models("*tiny_vit*"))
    name = name.lower()
    # Torchvision
    if name == "efficientnetv2_s":
        model = tv_models.efficientnet_v2_s(pretrained=pretrained)
    elif name == "convnext_tiny":
        model = tv_models.convnext_tiny(pretrained=pretrained)
    elif name == "mobilenetv3_large_100":
        model = tv_models.mobilenet_v3_large(pretrained=pretrained)
    else:
        # Timm models
        # timm_map = {
        #     "mobilevitv2_xs":    "mobilevitv2_xs",
        #     "mobilevitv2_s":     "mobilevitv2_s",
        #     "tiny_vit_21m_224":  "tiny_vit_21m_224",
        #     "repvgg_a2":         "repvgg_a2",
        # }
        timm_map = {
            "efficientnetv2_s": "efficientnetv2_s",
            "convnext_tiny": "convnext_tiny",  # as-is
            "mobilevitv2_xs": "mobilevit_xs",
            "mobilevitv2_s": "mobilevit_s",
            "mobilenetv3_large_100": "mobilenetv3_large_100",
            "tiny_vit_21m_224": "tiny_vit_21m_224",
            "repvgg_a2": "repvgg_a2",
            # … any other variants …
        }

        # convnext_tiny
        # mobilevitv2_xs
        # mobilevitv2_s
        # tiny_vit_21m_224
        # mobilenetv3_large_100
        # repvgg_a2
        # efficientnetv2_s

        if name in timm_map:
            model = timm.create_model(timm_map[name], pretrained=pretrained)
        else:
            raise ValueError(
                f"Model '{name}' is not supported. Available: "
                "efficientnetv2_s, convnext_tiny, mobilenetv3_large_100, "
                "mobilevitv2_xs, mobilevitv2_s, tiny_vit_21m_224, repvgg_a2"
            )

    # Replace classifier head if needed
    def _set_head(m):
        if hasattr(m, 'classifier'):
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = torch.nn.Linear(in_f, num_classes)
        elif hasattr(m, 'fc'):
            in_f = m.fc.in_features
            m.fc = torch.nn.Linear(in_f, num_classes)
        else:
            m.reset_classifier(num_classes=num_classes)

    # Always ensure correct num_classes
    _set_head(model)
    return model
