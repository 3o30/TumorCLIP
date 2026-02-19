"""
Project Constants Definition

This module centralizes all global constants used throughout the project.
It includes:

- Dataset class labels
- Professional multi-language medical prompts for CLIP-based models

"""

# ==================== Dataset Class Configuration ====================

# The order of this list determines the class index mapping.
CLASS_NAMES = [
    "Glioma",                    
    "Meningioma",               
    "NORMAL",                    
    "Neurocitoma",              
    "Outros Tipos de Lesões",   
    "Schwannoma" 
]

# Total number of classification categories
NUM_CLASSES = len(CLASS_NAMES)


# ==================== Professional Medical Prompts ====================
# Multi-language radiological descriptions used for CLIP text encoding.

# Languages included:
# - English (radiology-style reporting language)
# - Chinese
# - Portuguese

PROFESSIONAL_MEDICAL_PROMPTS = {
    "Glioma": [
        "MRI showing intra-axial glioma with infiltrative margins",
        "FLAIR hyperintense glioma in cerebral white matter", 
        "Ring-enhancing mass suspicious for high-grade glioma",
        "脑MRI提示胶质瘤，边界浸润，T2高信号",
        "Lesão intra-axial compatível com glioma na RM"
    ],

    "Meningioma": [
        "Extra-axial mass consistent with meningioma on MRI",
        "Dural-based lesion with homogeneous enhancement",
        "Convexity meningioma with dural tail sign",
        "颅内脑膜瘤，硬膜附着，均匀强化",
        "Meningioma extra-axial com realce homogêneo"
    ],

    "NORMAL": [
        "Normal brain MRI without pathological findings",
        "No evidence of mass lesion or abnormal enhancement",
        "Brain MRI shows normal anatomy and signal intensity",
        "正常脑MRI，未见异常信号或占位",
        "RM cerebral normal sem alterações patológicas"
    ],

    "Neurocitoma": [
        "Intraventricular mass consistent with neurocytoma",
        "Central neurocytoma in lateral ventricle",
        "侧脑室内神经细胞瘤，边界清楚",
        "Neurocitoma intraventricular com bordas bem definidas"
    ],

    "Outros Tipos de Lesões": [
        "Miscellaneous intracranial lesions on MRI",
        "Other types of cerebral pathological findings",
        "其他类型的颅内病变，MRI表现多样",
        "Outras lesões intracranianas de diferentes etiologias"
    ],

    "Schwannoma": [
        "Extra-axial schwannoma with cystic components",
        "Vestibular schwannoma in cerebellopontine angle",
        "小脑桥脑角神经鞘瘤，边界清楚",
        "Schwannoma do ângulo pontocerebelar com realce intenso"
    ],
}


# ==================== Utility Functions ====================

def get_class_index(class_name):
    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return -1


def get_class_name(index):

    if 0 <= index < NUM_CLASSES:
        return CLASS_NAMES[index]
    return None


def get_prompts_for_class(class_name):
    
    return PROFESSIONAL_MEDICAL_PROMPTS.get(class_name, [])
