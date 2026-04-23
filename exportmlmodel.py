import xgboost as xgb
import coremltools as ct

# 1. Load model XGBoost
# Pastikan menggunakan xgb.Booster() jika load dari .json
model = xgb.Booster()
model.load_model("salary_hike.json")

# 2. Konversi ke CoreML (Versi Terbaru)
# coremltools akan mendeteksi secara otomatis bahwa ini adalah XGBoost
coreml_model = ct.converters.xgboost.convert(
    model,
    feature_names=[
        'JobInvolvement',
        'JobLevel',
        'MonthlyIncome',
        'YearsAtCompany',
        'OverTime'
    ]
)

# 3. Simpan model
coreml_model.save("salary_hike.mlmodel")

print("Konversi berhasil tanpa error!")