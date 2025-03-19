import random

# List of universities from the document
universities = [
    "École Polytechnique Fédérale de Lausanne (EPFL)",
    "ETH Zurich (Swiss Federal Institute of Technology in Zurich)",
    "University of Zurich",
    "University of Basel",
    "University of Bern",
    "University of Geneva",
    "University of Lausanne",
    "Università della Svizzera italiana (USI)",
    "Lucerne School of Computer Science and Information Technology",
    "University of Fribourg",
    "University of Neuchâtel",
    "Zurich University of Applied Sciences (ZHAW)",
    "Bern University of Applied Sciences (BFH)",
    "University of Applied Sciences and Arts Northwestern Switzerland (FHNW)",
    "University of Applied Sciences and Arts Western Switzerland (HES-SO)",
    "Lucerne University of Applied Sciences and Arts",
    "Eastern Switzerland University of Applied Sciences (OST)",
    "University of Applied Sciences and Arts of Southern Switzerland (SUPSI)",
    "University of Applied Sciences of the Grisons (FHGR)",
    "Kalaidos University of Applied Sciences"
]

# List of people
people = ["Johan", "Nhat", "Thilo", "Wenxing"]

# Set a seed for reproducibility
random.seed(42)

# Shuffle the universities
random_universities = universities.copy()
random.shuffle(random_universities)

# Create a dictionary to store the assignments
assignments = {person: [] for person in people}

# Assign universities to people (5 universities each)
for i, university in enumerate(random_universities):
    person_index = i % len(people)
    assignments[people[person_index]].append(university)

# Print the assignments to console
for person, assigned_universities in assignments.items():
    print(f"\n{person}'s Universities:")
    for i, uni in enumerate(assigned_universities, 1):
        print(f"{i}. {uni}")

# Save the assignments to a text file
with open("university_assignments.txt", "w") as file:
    file.write("UNIVERSITY ASSIGNMENTS\n")
    file.write("=====================\n\n")
    for person, assigned_universities in assignments.items():
        file.write(f"{person}'s Universities:\n")
        for i, uni in enumerate(assigned_universities, 1):
            file.write(f"{i}. {uni}\n")
        file.write("\n")

    file.write("\nGenerated with random seed: 42")
