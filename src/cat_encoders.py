import category_encoders as ce

from cesamo.CesamoEncoder import CesamoEncoder


def set(cat_cols, encoder_list):
    """
    cat_cols: categorical columns in df
    encoder_list: encoders to apply
    """
    all_encoders_available = [
        {"encoder_name": "OrdinalEncoder", "encoder_obj": ce.OrdinalEncoder(cat_cols)},
        {"encoder_name": "OneHotEncoder", "encoder_obj": ce.OneHotEncoder(cat_cols)},
        {"encoder_name": "HelmertEncoder", "encoder_obj": ce.HelmertEncoder(cat_cols)},
        {
            "encoder_name": "BackwardDifferenceEncoder",
            "encoder_obj": ce.BackwardDifferenceEncoder(cat_cols),
        },
        {"encoder_name": "TargetEncoder", "encoder_obj": ce.TargetEncoder(cat_cols)},
        {"encoder_name": "SumEncoder", "encoder_obj": ce.SumEncoder(cat_cols)},
        {
            "encoder_name": "MEstimateEncoder",
            "encoder_obj": ce.MEstimateEncoder(cat_cols),
        },
        {
            "encoder_name": "LeaveOneOutEncoder",
            "encoder_obj": ce.LeaveOneOutEncoder(cat_cols),
        },
        {
            "encoder_name": "CatBoostEncoder",
            "encoder_obj": ce.CatBoostEncoder(cat_cols),
        },
        {
            "encoder_name": "JamesSteinEncoder",
            "encoder_obj": ce.JamesSteinEncoder(cat_cols),
        },
        {"encoder_name": "CesamoEncoder", "encoder_obj": CesamoEncoder(cat_cols)},
        {"encoder_name": "CountEncoder", "encoder_obj": ce.CountEncoder(cat_cols)},
        {"encoder_name": "BinaryEncoder", "encoder_obj": ce.BinaryEncoder(cat_cols)},
        {"encoder_name": "HashingEncoder", "encoder_obj": ce.HashingEncoder(cat_cols)},
    ]
    final_encoders = []
    for enc in all_encoders_available:
        if enc["encoder_name"] in encoder_list:
            final_encoders.append(enc)

    return final_encoders
