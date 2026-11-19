# Prediksi Churn Pelanggan Telekomunikasi dengan Machine Learning dan Explainable AI untuk Strategi Retensi Berbasis Data

Proyek akhir Praktikum Data Science 2026/2027, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta. Proyek yang sama juga menjadi bahan laporan UTS dan UAS mata kuliah teori Data Science.

## Ringkasan Proyek

**Masalah.** Perusahaan telekomunikasi belum punya cara sistematis untuk mengenali pelanggan yang berisiko berhenti berlangganan (churn) sebelum mereka pergi, sehingga program retensi cenderung massal, mahal, dan kurang tepat sasaran. Pada dataset yang dipakai, 1.869 dari 7.043 pelanggan (26,5%) berstatus churn, dengan porsi 30,5% dari total tagihan bulanan.

**Pertanyaan yang dijawab.**
1. Seberapa baik karakteristik layanan, akun, dan demografi pelanggan dapat memprediksi churn?
2. Faktor apa yang paling berpengaruh terhadap risiko churn dan dapat ditindaklanjuti?
3. Bagaimana skor risiko churn diubah menjadi keputusan retensi yang efisien secara biaya?

**Metode.** Klasifikasi biner (Churn = Yes atau No) dengan dua model dari materi praktikum: Random Forest (model utama) dan SVM (pembanding). Logistic Regression dan DummyClassifier ditambahkan hanya sebagai pembanding kewajaran, bukan kandidat model terpilih. Hyperparameter disetel dengan cross-validation pada data latih, dan threshold dipilih dari prediksi out-of-fold. Interpretasi memakai permutation importance (dengan uji stabilitas antar lipatan) dan SHAP.

**Metrik.** ROC-AUC, PR-AUC, recall, precision, dan F1 kelas churn, lift pada decile teratas, net benefit kampanye retensi, serta selisih recall antar kelompok (gender dan SeniorCitizen) sebagai pemeriksaan fairness. Accuracy hanya pelengkap karena kelas tidak seimbang.

## Anggota Kelompok dan Kontribusi

| No | Nama | NIM | Kontribusi |
| --- | --- | --- | --- |
| 1 | Wahyu Fahri Roisya | 123240052 | [grafik dan visualisasi, Cleaning dan perbandingan sebelum/sesudah, Analisis dan perbandingan] |
| 2 | Caesar Sabillah KPP | 123240056 | [EDA, Feature Engineering dan Selection,  perbaikan README dan requirements, Modelling RF] |
| 3 | Dzaky Muammar | 123240057 | [grafik dan visualisasi, Feature Engineering dan Selection, Modelling LR] |
| 4 | Rizky Rama Oktavian | 123240087 | [data gathering, Split data dan pipeline preprocessing, Modelling SVM] |

## Sumber Data

- **Dataset:** Telco Customer Churn, 7.043 baris dan 21 kolom (satu baris per pelanggan).
- **Asal:** data sampel IBM yang menggambarkan perusahaan telekomunikasi **fiktif**, dipublikasikan ulang di Kaggle oleh BlastChar.
- **Tautan:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Sumber asli:** https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
- **Tanggal akses:** 17 Oktober 2026.
- **Lisensi:** halaman Kaggle mencantumkan "Data files (c) Original Authors". Data dipakai hanya untuk tujuan akademik dengan atribusi kepada IBM. Berkas data disertakan di repository ini hanya untuk keperluan penilaian.
- **Deskripsi kolom:** tersedia pada data dictionary di bagian 2.6 notebook.

## Struktur Repository

```
.
+-- data/
|   +-- Telco-Customer-Churn.csv
+-- figures/                  # grafik hasil EDA dan evaluasi
+-- notebooks/
|   +-- Insight-EDA.ipynb     # alur lengkap: Business Understanding sampai Konklusi
+-- requirements.txt
+-- README.md
```

## Cara Menjalankan

1. Pastikan Python terpasang (proyek ini dikembangkan dengan Python 3.13.7). Disarankan memakai virtual environment.
2. Pasang dependensi. Versi library tercantum di `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
3. Pastikan dataset berada di `data/Telco-Customer-Churn.csv`.
4. Seluruh notebook berjalan sekitar 10 sampai 15 menit pada satu inti CPU (random search Random Forest dan grid SVM paling lama). Pada CPU multi-inti jauh lebih cepat karena `N_JOBS = -1`. Ubah `N_JOBS` menjadi 1 pada bagian 5.1 jika memori terbatas.
5. Buka folder `notebooks/` lalu jalankan notebook berurutan dari sel pertama sampai terakhir. Path pada notebook relatif terhadap folder `notebooks/` (`../data/` dan `../figures/`), sehingga notebook harus dijalankan dengan folder kerja `notebooks/`.
6. Grafik tersimpan otomatis di folder `figures/`. Seluruh hasil dapat direproduksi karena `RANDOM_STATE = 42`.

## Ringkasan Cleaning dan Preprocessing

- **Cleaning.** `TotalCharges` diubah dari teks ke angka dan 11 nilai kosong tersembunyi (berupa spasi, semuanya pelanggan bertenure 0) diisi 0. Kategori "No internet service" dan "No phone service" disatukan ke "No", `SeniorCitizen` diseragamkan menjadi Yes/No, dan `customerID` dibuang. Tidak ada baris yang dibuang.
- **Pembagian data.** 80% data latih (5.634 baris, 1.495 churn) dan 20% data uji (1.409 baris, 374 churn) secara terstratifikasi, dengan proporsi churn sama di kedua bagian (26,54%). Pembagian dilakukan sebelum Feature Selection agar data uji tidak memengaruhi pemilihan fitur.
- **Fitur.** Setelah Feature Selection (bagian 4.3) ada 18 fitur (3 numerik dan 15 kategorikal), termasuk fitur turunan `jumlah_layanan_tambahan`. `gender` tidak dipakai sebagai fitur dan disimpan untuk audit fairness. Pada tahap Modelling (bagian 5.3) dilakukan ablation dengan aturan keputusan tertulis: `TotalCharges` dipakai kembali (PR-AUC validasi naik 0,005), fitur demografis (`SeniorCitizen`, `Partner`, `Dependents`) dibuang karena tidak menurunkan performa, dan empat fitur turunan tambahan tidak dipakai karena tidak membantu. **Model akhir memakai 16 fitur** (4 numerik dan 12 kategorikal).
- **Pipeline.** One-hot encoding untuk fitur kategorikal, dan standardisasi fitur numerik hanya untuk SVM. Semua transformasi dimasukkan ke dalam pipeline sehingga hanya di-fit pada data latih.

## Target Metrik dan Hasil

Target ditetapkan sebelum modelling dan tidak diubah setelah melihat hasil. Target berlaku untuk model terpilih (Random Forest tuned), dengan threshold 0,406 yang dipilih dari prediksi out-of-fold pada data latih. Metrik berikut dihitung pada data uji (1.409 pelanggan, 374 churn).

| Metrik | Target | Hasil | Tercapai |
| --- | --- | --- | --- |
| ROC-AUC | >= 0,80 | 0,846 (CI 95% 0,824 sampai 0,867) | Ya |
| PR-AUC (baseline 0,265) | >= 0,55 | 0,654 (CI 95% 0,602 sampai 0,703) | Ya |
| Recall kelas churn | >= 0,65 | 0,824 | Ya |
| Precision kelas churn | >= 0,40 | 0,493 | Ya |
| Lift decile teratas | >= 2,0 | 2,89 | Ya |
| Net benefit | Positif dan lebih tinggi daripada menghubungi semua pelanggan | Terpenuhi pada tiga skenario biaya jika threshold disesuaikan per skenario. Dengan satu threshold tunggal hanya terpenuhi pada skenario dasar | Ya (dengan catatan) |
| Korelasi peringkat fitur antar fold | >= 0,70 | 0,87 (terendah 0,74) | Ya |
| Selisih recall antar kelompok | <= 20 poin persentase | Gender 2,1 poin, SeniorCitizen 14,2 poin | Ya |

**Perbandingan model pada data uji** (selisih dilaporkan dengan interval kepercayaan bootstrap 95%):

| Model | ROC-AUC | PR-AUC |
| --- | --- | --- |
| RF baseline (parameter bawaan, fitur awal) | 0,816 | 0,598 |
| **RF tuned (model terpilih)** | **0,846** | **0,654** |
| SVM baseline | 0,820 | 0,592 |
| SVM tuned | 0,837 | 0,637 |
| Logistic Regression (pembanding kewajaran) | 0,842 | 0,636 |

**Temuan utama**
- Tuning memperbaiki Random Forest secara nyata (ROC-AUC +0,030 dengan CI [0,018, 0,043]), terutama karena Random Forest bawaan menghafal data latih (PR-AUC latih 0,9999 vs 0,598 pada uji).
- Random Forest **setara** dengan SVM tuned dan Logistic Regression, bukan lebih unggul: semua selisih interval kepercayaannya mencakup atau menempel pada nol. ROC-AUC sekitar 0,84 tampaknya batas yang wajar pada dataset ini.
- Faktor utama: jenis kontrak, jenis layanan internet, dan lama berlangganan (hasil permutation importance dan SHAP, bersifat asosiasi).
- Pada skenario biaya dasar, menghubungi 44% pelanggan teratas menangkap 82% churn dengan net benefit sekitar 2,4 kali lipat menghubungi semua pelanggan.
- Repeated CV 5x5 pada 7.043 baris (bagian 6.9) memperkuat hal itu: RF vs Logistic Regression selisih ROC-AUC +0,001 (tidak signifikan), RF vs SVM +0,005 (di batas signifikansi, tidak bermakna secara praktis). Learning curve sudah mendatar, sehingga batasnya ada pada informasi di fitur, bukan pada model atau jumlah data.
- Kelemahan: churn pada pelanggan berkontrak satu atau dua tahun hampir tidak terdeteksi (36 dari 45 churner uji terlewat).

## Status Pengerjaan

**Batas pengumpulan 30 Oktober 2026**
- [x] Pemilihan metode: Klasifikasi
- [x] Business Understanding
- [x] Data Gathering
- [x] EDA
- [x] Cleaning dan Preprocessing
- [x] Feature Selection

**Batas pengumpulan akhir 22 November 2026**
- [x] Modelling
- [x] Evaluasi
- [x] Konklusi

Presentasi: 23 November sampai 4 Desember 2026.

## Batasan

- Data berupa satu snapshot per pelanggan tanpa riwayat waktu, sehingga dinamika perubahan perilaku tidak dapat dimodelkan.
- Perusahaan dalam dataset fiktif dan tanpa informasi wilayah, sehingga hasil tidak dapat langsung digeneralisasi ke operator di Indonesia.
- Tidak ada data biaya akuisisi, margin, atau customer lifetime value, sehingga analisis cost-benefit memakai asumsi.
- Hubungan antar variabel bersifat asosiatif, bukan sebab-akibat.
- Pemodelan dibatasi pada Random Forest dan SVM sesuai materi praktikum. Logistic Regression hanya dipakai sebagai pembanding kewajaran.
- Data uji hanya berisi 374 pelanggan churn, sehingga perbedaan ROC-AUC di bawah sekitar 0,04 antar model tidak dapat dibedakan dari kebetulan sampel.
- Net benefit memakai tiga skenario asumsi biaya dan peluang penerimaan, bukan data nyata.