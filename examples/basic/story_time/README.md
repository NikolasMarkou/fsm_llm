# Three Little Pigs Interactive Story

This example demonstrates using LLM-FSM to create an interactive storytelling experience based on the classic tale of the Three Little Pigs.

## Purpose

This example shows how to:
- Create an interactive narrative with user input at key story moments
- Incorporate user suggestions directly into the storyline
- Reference previous user contributions for narrative continuity
- Use the persona feature to maintain a consistent narrative voice
- Build a linear story progression with interactive elements

## FSM Structure

This example implements an interactive story with the following states:
1. **introduction**: Sets the scene and introduces the three little pigs
2. **first_pig_house**: Describes the first pig building a house of straw
3. **wolf_attacks_straw**: Narrates the wolf destroying the straw house
4. **second_pig_house**: Describes the second pig building a house of sticks
5. **wolf_attacks_sticks**: Narrates the wolf destroying the stick house
6. **third_pig_house**: Describes the third pig building a brick house
7. **wolf_attacks_brick**: Narrates the wolf failing to blow down the brick house
8. **wolf_tries_chimney**: Describes the wolf attempting to enter through the chimney
9. **conclusion**: Wraps up the story with the wolf's defeat and the moral
10. **feedback**: Gets user feedback on the storytelling experience
11. **end**: Concludes the interaction gracefully

## How to Run

1. Make sure you have LLM-FSM installed:
```bash
pip install llm-fsm
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the example:
```bash
python simple_story.py
```

## Expected Output

```
============================================================
        📚 THREE LITTLE PIGS INTERACTIVE STORY 📚        
============================================================

Storyteller: Welcome to the interactive story of the Three Little Pigs! Once upon a time, there were three little pigs who left home to seek their fortune. Each would soon face the challenge of building a home to protect themselves from the dangers of the world. Are you ready to begin this classic tale?

You: Yes, I'm ready

Storyteller: Excellent! Our tale begins with the first little pig. This young pig, eager to quickly finish his home and enjoy his freedom, decided to build his house out of straw. He gathered bundles of golden straw from a nearby field and hastily constructed a simple dwelling in just one day.

The house of straw stood lightly in the meadow, its golden walls gleaming in the sunshine. The first little pig was quite pleased with himself for completing his home so quickly.

What do you think about the first pig's decision to build his house out of straw?

You: I think it's not a wise choice because straw is too weak

Storyteller: Indeed, your wisdom mirrors that of many who have heard this tale before! The first little pig's haste would soon prove costly.

Not long after the straw house was completed, a big bad wolf came prowling through the area. With his keen nose, he detected the scent of pig and followed it straight to the straw house.

The wolf approached the door and called out in a deceptively friendly voice, "Little pig, little pig, let me come in!"

From inside his fragile dwelling, the pig replied, "No, no, by the hair of my chinny chin chin, I will not let you in!"

Enraged by this defiance, the wolf growled, "Then I'll huff, and I'll puff, and I'll blow your house in!"

Drawing a deep breath, the wolf exhaled with tremendous force. Just as you wisely anticipated, the house of straw offered little resistance. It collapsed immediately, scattering straw in all directions. The terrified pig barely escaped, running as fast as his little trotters could carry him to his brother's house.

You: Where is his brother's house?

Storyteller: The first little pig ran frantically across fields and through woods until he reached his brother's new home not far away.

This second little pig had decided to build his house out of sticks. He had spent several days gathering sturdy branches and twigs from the forest, binding them together to create walls and a roof. The house of sticks stood taller than the straw house, with walls that appeared more substantial.

When his frightened brother arrived, telling tales of the wolf's attack, the second pig welcomed him inside confidently. "Don't worry, brother," he assured. "My house is much stronger than yours. The wolf won't be able to blow this one down!"

The two pigs settled in, believing they were safe behind the wooden walls of the stick house.

Do you think this house of sticks will be strong enough to protect the pigs from the wolf?

You: No, the sticks aren't secured well enough to withstand the wolf's breath

Storyteller: Your assessment shows remarkable foresight! The second pig's confidence would indeed prove misplaced.

Before long, the wolf tracked the first pig's scent to the house of sticks. Seeing this somewhat sturdier dwelling, he approached with determination. Once again, he called out, "Little pigs, little pigs, let me come in!"

From inside, the second pig replied boldly, "No, no, by the hair of our chinny chin chins, we will not let you in!"

The wolf's eyes narrowed as he threatened, "Then I'll huff, and I'll puff, and I'll blow your house in!"

Taking an even deeper breath than before, the wolf unleashed a mighty gust against the house of sticks. As you correctly predicted, despite being stronger than straw, the sticks were not secured well enough. The walls shuddered, then broke apart as the house collapsed around the terrified pigs.

Once again, the pigs managed to escape just in time, racing away with the wolf in pursuit. They ran until they reached the home of their third and wisest brother.

You: What kind of house did the third pig build?

Storyteller: The third pig, the wisest of the brothers, had built his house entirely of bricks. Unlike his siblings, he had spent many weeks on his construction, carefully laying each brick with proper mortar, ensuring every wall was sturdy and well-built. He had installed a proper door with a strong lock, windows with shutters, and even constructed a stone chimney for warmth.

When his two brothers arrived, breathless and frightened, he quickly ushered them inside and secured the heavy door behind them. They told him about the wolf and how easily he had destroyed both the straw and stick houses.

The third pig remained calm. "My house is built of solid brick," he assured them. "It's properly constructed with a strong foundation. We'll be safe here."

Soon, they heard heavy footsteps outside and knew the wolf had found them once again.

Do you think the wolf will be able to blow down this brick house?

You: No, the brick house is too solid and well-constructed for the wolf to blow down

Storyteller: Your confidence in the third pig's craftsmanship is well-placed! The brick house was indeed built to withstand far more than a wolf's breath.

When the wolf arrived, he circled the brick house, studying it with growing frustration. Still determined to catch the pigs, he approached the door and called out, "Little pigs, little pigs, let me come in!"

From within their secure shelter, the third pig responded firmly, "No, no, by the hair of our chinny chin chins, we will not let you in!"

The wolf snarled, "Then I'll huff, and I'll puff, and I'll blow your house in!"

He took the deepest breath yet, his chest swelling to enormous size, and blew with all his might against the brick house. But just as you predicted, the solid walls didn't even tremble. The wolf tried again and again until he was exhausted, but the brick house stood strong and unmoving.

Frustrated but not defeated, the wolf began looking for another way in. His eyes traveled upward, and he spotted the chimney. With a wicked grin, he decided to climb onto the roof and enter through this opening.

What do you think the three pigs should do now that the wolf is planning to come down the chimney?

You: They should light a fire and put a big pot of water to boil under the chimney

Storyteller: That's an excellent strategy! The third pig, being quite clever, had the same thought.

Seeing the wolf's shadow moving across the roof toward the chimney, the third pig quickly sprang into action. "Brothers, help me quickly!" he called.

Together, they built a roaring fire in the fireplace and hung a large iron cauldron over the flames. They filled it with water and watched as it began to heat and bubble.

Above them, they could hear the wolf's claws scraping against the chimney as he prepared to descend. The pigs stepped back from the fireplace, waiting anxiously as soot began to fall into the bubbling water.

Suddenly, with a scraping sound, the wolf came sliding down the chimney—directly into the cauldron of boiling water! With a tremendous howl of pain, he leaped out, shot back up the chimney, and fled across the countryside, never to return to trouble the three little pigs again.

You: That's a great ending! I enjoyed the story.

Storyteller: Thank you for joining me on this interactive journey through the tale of the Three Little Pigs! I'm delighted that you enjoyed the story.

The three pigs lived happily together in the brick house for some time afterward. The younger pigs had learned valuable lessons about taking time to build things properly and the importance of planning for the future rather than seeking quick and easy solutions.

The tale of the three little pigs has been told for generations, teaching children about the virtues of hard work, patience, and thinking ahead. Through your participation, you've helped bring this classic story to life in a new way.

How did you find this interactive storytelling experience? Would you be interested in experiencing other classic tales in this format in the future?

You: Yes, I'd love to experience more interactive stories like this one!

Storyteller: I'm thrilled to hear that! Your enthusiasm for interactive storytelling warms my heart. There are indeed many more tales waiting to be explored—from Goldilocks and the Three Bears to Jack and the Beanstalk, from Little Red Riding Hood to Hansel and Gretel.

Each story offers its own adventures and moments where your input could shape the narrative in meaningful ways. Your suggestions and thoughts have made this telling of the Three Little Pigs particularly special.

Thank you for your participation and for keeping the ancient art of storytelling alive through your engagement. Until our next tale, I bid you farewell and wish you many more adventures in the world of stories!

============================================================
                  📖 STORY COMPLETE! 📖                  
============================================================

Your Journey Through the Story:
• Opinion Straw: I think it's not a wise choice because straw is too weak
• Opinion Sticks: No, the sticks aren't secured well enough to withstand the wolf's breath
• Prediction: No, the brick house is too solid and well-constructed for the wolf to blow down
• Suggestion: They should light a fire and put a big pot of water to boil under the chimney
• Feedback: Yes, I'd love to experience more interactive stories like this one!

Thank you for experiencing this interactive tale!
```

## Learning Points

This example demonstrates:
- How to structure a linear narrative with interactive elements
- Using `required_context_keys` to ensure user input is collected at key story points
- How to reference previous user contributions for narrative continuity
- Implementing the persona feature for consistent storytelling voice
- Creating a satisfying story arc with proper resolution
- How to track and summarize user contributions throughout the experience
- Building an engaging user experience with minimal implementation code