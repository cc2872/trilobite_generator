"""GBIF datasets that match the names Trilobita / Agnostida (or a trilobite genus) but hold living organisms: eDNA
samples, herbaria, seaweed and insect collections whose genera share a name with a trilobite genus. Trilobites are
extinct, so none of these records can be one. Checked by hand on 14 Nov 2026 (record counts at that date)."""
NOT_FOSSIL = {
    "9fc74265-28a7-4d62-97ea-33eb5588562e",  #  76 Australian Microbiome 18S (eukaryote) aquatic samples (eDNA)
    "e3dca178-5cef-456e-b3b0-6dc4b02bbd16",  #  40 University and Jepson Herbaria, Algae (Brongniartella, a red alga)
    "14f3151a-e95d-493c-a40d-d9938ef62954",  #  19 CAS Entomology (Conocephalus, a katydid)
    "7401f23d-e962-4828-8646-dc4d8cfc2394",  #  11 UCT South Africa Seaweeds (Brongniartella)
    "bd14614c-4d72-4763-8710-1cea514311cc",  #   7 EUKARYOME rRNA reference database (eDNA)
    "5d283bb6-64dd-4626-8b3b-a4e8db5415c3",  #   5 Essig Museum of Entomology
    "15738fee-78b1-4751-bcc1-cbf7139c68de",  #   4 CANB AVH herbarium data
    "5a6538b8-e35c-46c4-8978-6858a657a75f",  #   2 Cornell Plant Pathology Herbarium (Neophillipsia)
    "42301ebf-d36a-4e4f-a061-d37d778c2365",  #   2 Tasmanian Herbarium AVH data
    "1a9d08bb-9e2c-4540-8f7f-3205846b6ddc",  #   1 UTEP Plants (Arctos)
    "9a228813-6288-4aba-a8f8-480e425b5f62",  #   1 Baltic Sea 18S metabarcoding (eDNA)
    "857bce66-f762-11e1-a439-00145eb45e9a",  #   1 ZOBODAT zoological-botanical database
    "d415c253-4d61-4459-9d25-4015b9084fb0",  #   1 New York Botanical Garden Herbarium
    "e2980e63-d152-4219-8c1e-0ffdef3ea6aa",  #   1 Auckland Museum NZ Marine Collection (recent marine)
    "f519367d-6b9d-411c-b319-99424741e7de",  #   1 insect occurrence data, multiple projects
    "4443b1f2-231b-4e60-b343-8c93fba93982",  #   1 Herbier de Nouvelle-Caledonie (Hibbertia, a plant)
}
