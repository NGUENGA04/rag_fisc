import unittest

from moteur_rag import MoteurRAG


class MemoireRagTests(unittest.TestCase):
    def setUp(self):
        self.moteur = MoteurRAG.__new__(MoteurRAG)
        self.moteur.memoire_conversations = {}
        self.moteur.max_tours_memoire = 4

    def test_ajouter_et_recuperer_historique(self):
        self.moteur.ajouter_echange("user-1", "Bonjour", "Bonjour !")
        self.moteur.ajouter_echange("user-1", "Quel est le taux de TVA ?", "Le taux est de 19,25 %")

        contexte = self.moteur._construire_contexte_memoire("user-1")

        self.assertIn("Bonjour", contexte)
        self.assertIn("Quel est le taux de TVA ?", contexte)
        self.assertIn("Le taux est de 19,25 %", contexte)


if __name__ == "__main__":
    unittest.main()
