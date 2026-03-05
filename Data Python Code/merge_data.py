import pandas as pd

# 1. โหลดไฟล์
df_old = pd.read_csv('CSV Files/sst_dhw_FINAL_2023_2024.csv')
df_new = pd.read_csv('CSV Files/sst_dhw_2023_2024_2025_combined.csv')

# 2. รวมไฟล์
df_all = pd.concat([df_old, df_new], ignore_index=True)

# 3. ตรวจหาคอลัมน์ "วันที่" (ลองหาทั้ง date, Date, DATE)
date_col = None
for col in df_all.columns:
    if col.lower() == 'date':
        date_col = col
        break

if date_col:
    # แปลงเป็น datetime และเรียงลำดับ
    df_all[date_col] = pd.to_datetime(df_all[date_col])
    df_all = df_all.sort_values(by=date_col)
    print(f"จัดเรียงข้อมูลตามคอลัมน์: '{date_col}' เรียบร้อย")
else:
    print("หาคอลัมน์ 'date' ไม่เจอ! คอลัมน์ที่มีคือ:", list(df_all.columns))

# 4. ลบข้อมูลซ้ำและบันทึก
df_all = df_all.drop_duplicates()
df_all.to_csv('CSV Files/sst_dhw_FINAL_2023_2025_complete.csv', index=False)

print("บันทึกไฟล์ใหม่สำเร็จ: sst_dhw_FINAL_2023_2025_complete.csv")