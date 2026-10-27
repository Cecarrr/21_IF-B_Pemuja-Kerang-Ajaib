# Prediksi Churn Pelanggan Telekomunikasi dengan Machine Learning dan Explainable AI untuk Strategi Retensi Berbasis Data

Proyek akhir Praktikum Data Science 2026/2027, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta. Proyek yang sama juga menjadi bahan laporan UTS dan UAS mata kuliah teori Data Science.

## Ringkasan Proyek

**Masalah.** Perusahaan telekomunikasi belum punya cara sistematis untuk mengenali pelanggan yang berisiko berhenti berlangganan (churn) sebelum mereka pergi, sehingga program retensi cenderung massal, mahal, dan kurang tepat sasaran. Pada dataset yang dipakai, 1.869 dari 7.043 pelanggan (26,5%) berstatus churn, dengan porsi 30,5% dari total tagihan bulanan.

**Pertanyaan yang dijawab.**
1. Seberapa baik karakteristik layanan, akun, dan demografi pelanggan dapat memprediksi churn?
2. Faktor apa yang paling berpengaruh terhadap risiko churn dan dapat ditindaklanjuti?
3. Bagaimana skor risiko churn diubah menjadi keputusan retensi yang efisien secara biaya?

**Metode.** Klasifikasi biner (Churn = Yes atau No) dengan dua model dari materi praktikum: Random Forest (model utama) dan SVM (pembanding). Interpretasi menggunakan SHAP dan feature importance Random Forest.

**Metrik.** ROC-AUC, PR-AUC, recall, precision, dan F1 kelas churn, lift pada decile teratas, net benefit kampanye retensi, serta selisih recall antar kelompok (gender dan SeniorCitizen) sebagai pemeriksaan fairness. Accuracy hanya pelengkap karena kelas tidak seimbang.

## Anggota Kelompok dan Kontribusi

| No | Nama | NIM | Kontribusi |
| --- | --- | --- | --- |
| 1 | Wahyu Fahri Roisya | 123240052 | [grafik dan visualisasi] |
| 2 | Caesar Sabillah KPP | 123240056 | [EDA] |
| 3 | Dzaky Muammar | 123240057 | [grafik dan visualisasi] |
| 4 | Rizky Rama Oktavian | 123240087 | [data gathering] |

## Sumber Data

- **Dataset:** Telco Customer Churn, 7.043 baris dan 21 kolom (satu baris per pelanggan).
- **Asal:** data sampel IBM yang menggambarkan perusahaan telekomunikasi **fiktif**, dipublikasikan ulang di Kaggle oleh BlastChar.
- **Tautan:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Sumber asli:** https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
- **Lisensi:** halaman Kaggle mencantumkan "Data files © Original Authors". Data dipakai hanya untuk tujuan akademik dengan atribusi kepada IBM. Berkas data disertakan di repository ini hanya untuk keperluan penilaian.
- **Catatan keabsahan:** karena datanya sampel fiktif, penggunaannya sudah dikonfirmasi dan diizinkan oleh asisten lab praktikum ([isi nama asisten dan tanggal]).
- **Deskripsi kolom:** tersedia pada data dictionary di laporan UTS dan pada tahap Data Gathering di notebook.

## Struktur Repository

```
.
├── data/
│   └── Telco-Customer-Churn.csv
├── figures/                  # grafik hasil EDA dan evaluasi
├── notebooks/
│   └── Insight-EDA.ipynb     # [tambahkan notebook modelling dan evaluasi]
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Pastikan Python terpasang (proyek ini dikembangkan dengan Python 3.13.7).
2. Pasang dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Pastikan dataset berada di `data/Telco-Customer-Churn.csv`.
4. Buka folder `notebooks/` lalu jalankan notebook berurutan dari sel pertama sampai terakhir.
5. Grafik tersimpan otomatis di folder `figures/`. Path data dan folder grafik dihitung dari root repository, jadi notebook bisa dijalankan dari folder `notebooks/`.

## Target Metrik dan Hasil

Target ditetapkan konservatif dan berlaku untuk model terpilih.

| Metrik | Target | Hasil |
| --- | --- | --- |
| ROC-AUC | ≥ 0,80 | [isi setelah evaluasi] |
| PR-AUC (baseline 0,265) | ≥ 0,55 | [isi setelah evaluasi] |
| Recall kelas churn | ≥ 0,65 | [isi setelah evaluasi] |
| Precision kelas churn | ≥ 0,40 | [isi setelah evaluasi] |
| Lift decile teratas | ≥ 2,0 | [isi setelah evaluasi] |
| Net benefit | Positif dan lebih tinggi daripada menghubungi semua pelanggan | [isi setelah evaluasi] |
| Korelasi peringkat SHAP antar fold | ≥ 0,70 | [isi setelah evaluasi] |
| Selisih recall antar kelompok | ≤ 20 poin persentase | [isi setelah evaluasi] |

## Status Pengerjaan

**Batas pengumpulan 30 Oktober 2026**
- [✓] Business Understanding
- [✓] Data Gathering
- [✓] EDA
- [ ] Cleaning dan Preprocessing

**Batas pengumpulan akhir 22 November 2026**
- [ ] Modelling
- [ ] Evaluasi
- [ ] Konklusi

Presentasi: 23 November sampai 4 Desember 2026.

## Batasan

- Data berupa satu snapshot per pelanggan tanpa riwayat waktu, sehingga dinamika perubahan perilaku tidak dapat dimodelkan.
- Perusahaan dalam dataset fiktif dan tanpa informasi wilayah, sehingga hasil tidak dapat langsung digeneralisasi ke operator di Indonesia.
- Tidak ada data biaya akuisisi, margin, atau customer lifetime value, sehingga analisis cost-benefit memakai asumsi.
- Hubungan antar variabel bersifat asosiatif, bukan sebab-akibat.
- Pemodelan dibatasi pada Random Forest dan SVM sesuai materi praktikum.
