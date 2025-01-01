craftax_classic_game_rules = {
    "World Information": {
        "Actions with Requirements": {
            "Movement": {
                "left": {"Requirements": "Flat ground west of the agent."},
                "right": {"Requirements": "Flat ground east of the agent."},
                "up": {"Requirements": "Flat ground north of the agent."},
                "down": {"Requirements": "Flat ground south of the agent."},
            },
            "Basic Actions": {
                "do": {
                    "Requirements": "Facing creature or material; have necessary tool."
                },
                "sleep": {"Requirements": "Energy level is below maximum."},
                "noop": {"Requirements": "Always applicable. (Means no operation.)"},
            },
            "Crafting and Building": {
                "place_stone": {"Requirements": "Stone in inventory."},
                "place_table": {"Requirements": "Wood in inventory."},
                "place_furnace": {"Requirements": "Stone in inventory."},
                "place_plant": {"Requirements": "Sapling in inventory."},
                "make_wood_pickaxe": {
                    "Requirements": "Nearby table; wood in inventory."
                },
                "make_stone_pickaxe": {
                    "Requirements": "Nearby table; wood, stone in inventory."
                },
                "make_iron_pickaxe": {
                    "Requirements": "Nearby table, furnace; wood, coal, iron in inventory."
                },
                "make_wood_sword": {"Requirements": "Nearby table; wood in inventory."},
                "make_stone_sword": {
                    "Requirements": "Nearby table; wood, stone in inventory."
                },
                "make_iron_sword": {
                    "Requirements": "Nearby table, furnace; wood, coal, iron in inventory."
                },
            },
        },
        "Hierarchy of Goals": {
            "Collect Wood": {
                "place_table": {
                    "make_wood_pickaxe": {
                        "Collect Stone": {
                            "place_stone": {},
                            "make_stone_pickaxe": {
                                "Collect Iron": {
                                    "make_iron_pickaxe": {
                                        "Collect Diamond": {},
                                        "place_furnace": {
                                            "make_iron_sword": {},
                                            "make_iron_pickaxe": {},
                                        },
                                    }
                                },
                                "Collect Coal": {},
                            },
                        }
                    },
                    "make_wood_sword": {},
                }
            },
            "Eat Cow": {},
            "Collect Sapling": {"place_plant": {"Eat Plant": {}}},
            "Collect Drink": {},
            "Defeat Zombie": {},
            "Defeat Skeleton": {},
            "Wake Up": {},
        },
    }
}
craftax_classic_action_dict = {
    "noop": 0,
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}

craftax_full_game_rules = {
    "World Information": {
        "Actions with Requirements": {
            "Movement": {
                "left": {"Requirements": "Flat ground west of the agent."},
                "right": {"Requirements": "Flat ground east of the agent."},
                "up": {"Requirements": "Flat ground north of the agent."},
                "down": {"Requirements": "Flat ground south of the agent."},
            },
            "Basic Actions": {
                "do": {
                    "Requirements": "Facing creature or material; have necessary tool."
                },
                "sleep": {"Requirements": "Energy level is below maximum."},
                "noop": {"Requirements": "Always applicable. (Means no operation.)"},
            },
            "Crafting and Building": {
                "place_stone": {
                    "Requirements": "Have stone in inventory; target location is empty or water."
                },
                "place_table": {
                    "Requirements": "Have 2 wood in inventory; target location is empty."
                },
                "place_furnace": {
                    "Requirements": "Have stone in inventory; target location is empty."
                },
                "place_torch": {
                    "Requirements": "Have torch in inventory; target location is valid for torch placement."
                },
                "place_plant": {
                    "Requirements": "Have sapling in inventory; target location is grass."
                },
                "make_wood_pickaxe": {
                    "Requirements": "At crafting table; have 1 wood in inventory."
                },
                "make_stone_pickaxe": {
                    "Requirements": "At crafting table; have 1 wood and 1 stone in inventory."
                },
                "make_iron_pickaxe": {
                    "Requirements": "At crafting table and furnace; have 1 wood, 1 stone, 1 iron, and 1 coal in inventory."
                },
                "make_diamond_pickaxe": {
                    "Requirements": "At crafting table; have 1 wood and 3 diamonds in inventory."
                },
                "make_wood_sword": {
                    "Requirements": "At crafting table; have 1 wood in inventory."
                },
                "make_stone_sword": {
                    "Requirements": "At crafting table; have 1 wood and 1 stone in inventory."
                },
                "make_iron_sword": {
                    "Requirements": "At crafting table and furnace; have 1 wood, 1 stone, 1 iron, and 1 coal in inventory."
                },
                "make_diamond_sword": {
                    "Requirements": "At crafting table; have 1 wood and 2 diamonds in inventory."
                },
                "make_iron_armour": {
                    "Requirements": "At crafting table and furnace; have 3 iron and 3 coal in inventory."
                },
                "make_diamond_armour": {
                    "Requirements": "At crafting table; have 3 diamonds in inventory."
                },
                "make_arrow": {
                    "Requirements": "At crafting table; have 1 wood and 1 stone in inventory."
                },
                "make_torch": {
                    "Requirements": "At crafting table; have 1 wood and 1 coal in inventory."
                },
            },
            "Combat and Tools": {
                "shoot_arrow": {"Requirements": "Have bow and arrows in inventory."},
                "cast_fireball": {
                    "Requirements": "Have learned fireball spell; have at least 2 mana."
                },
                "cast_iceball": {
                    "Requirements": "Have learned iceball spell; have at least 2 mana."
                },
                "enchant_bow": {
                    "Requirements": "At enchantment table; have bow, 1 gem (ruby/sapphire), and 9 mana."
                },
                "enchant_sword": {
                    "Requirements": "At enchantment table; have sword, 1 gem (ruby/sapphire), and 9 mana."
                },
                "enchant_armour": {
                    "Requirements": "At enchantment table; have armour, 1 gem (ruby/sapphire), and 9 mana."
                },
            },
            "Consumables": {
                "drink_potion_red": {"Requirements": "Have red potion in inventory."},
                "drink_potion_green": {
                    "Requirements": "Have green potion in inventory."
                },
                "drink_potion_blue": {"Requirements": "Have blue potion in inventory."},
                "drink_potion_pink": {"Requirements": "Have pink potion in inventory."},
                "drink_potion_cyan": {"Requirements": "Have cyan potion in inventory."},
                "drink_potion_yellow": {
                    "Requirements": "Have yellow potion in inventory."
                },
                "read_book": {"Requirements": "Have book in inventory."},
            },
            "Level Navigation": {
                "descend": {
                    "Requirements": "On down ladder; current level cleared of monsters."
                },
                "ascend": {"Requirements": "On up ladder."},
            },
            "Character Development": {
                "level_up_dexterity": {
                    "Requirements": "Have at least 1 XP; dexterity below maximum."
                },
                "level_up_strength": {
                    "Requirements": "Have at least 1 XP; strength below maximum."
                },
                "level_up_intelligence": {
                    "Requirements": "Have at least 1 XP; intelligence below maximum."
                },
            },
        }
    },
    "Hierarchy of Goals": {
        "Collect Wood": {
            "place_table": {
                "make_wood_pickaxe": {
                    "Collect Stone": {
                        "place_stone": {},
                        "make_stone_pickaxe": {
                            "Collect Iron": {
                                "make_iron_pickaxe": {
                                    "Collect Diamond": {
                                        "make_diamond_pickaxe": {},
                                        "make_diamond_sword": {},
                                        "make_diamond_armour": {},
                                    },
                                    "place_furnace": {
                                        "make_iron_sword": {},
                                        "make_iron_armour": {},
                                    },
                                }
                            },
                            "Collect Coal": {},
                        },
                    }
                },
                "make_wood_sword": {},
            }
        },
        "Eat Cow": {},
        "Collect Sapling": {"place_plant": {"Eat Plant": {}}},
        "Collect Drink": {},
        "Defeat Zombie": {},
        "Defeat Skeleton": {},
        "Wake Up": {},
        "Make Arrow": {},
        "Make Torch": {"Place Torch": {}},
        "Enter Gnomish Mines": {"Defeat Gnome Warrior": {}, "Defeat Gnome Archer": {}},
        "Enter Dungeon": {},
        "Enter Sewers": {},
        "Enter Vault": {},
        "Enter Troll Mines": {"Defeat Troll": {}},
        "Enter Fire Realm": {"Defeat Fire Elemental": {}, "Defeat Pigman": {}},
        "Enter Ice Realm": {"Defeat Frost Troll": {}, "Defeat Ice Elemental": {}},
        "Enter Graveyard": {"Damage Necromancer": {"Defeat Necromancer": {}}},
        "Defeat Orc Soldier": {},
        "Defeat Orc Mage": {},
        "Defeat Lizard": {},
        "Defeat Kobold": {},
        "Defeat Deep Thing": {},
        "Eat Bat": {},
        "Eat Snail": {},
        "Find Bow": {"Fire Bow": {}},
        "Collect Sapphire": {"Learn Iceball": {"Cast Iceball": {}}},
        "Collect Ruby": {"Learn Fireball": {"Cast Fireball": {}}},
        "Open Chest": {},
        "Drink Potion": {},
        "Enchant Sword": {},
        "Enchant Armour": {},
        "Defeat Knight": {},
        "Defeat Archer": {},
    },
}
craftax_full_action_dict = {
    "noop": 0,
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
    "rest": 17,
    "descend": 18,
    "ascend": 19,
    "make_diamond_pickaxe": 20,
    "make_diamond_sword": 21,
    "make_iron_armour": 22,
    "make_diamond_armour": 23,
    "shoot_arrow": 24,
    "make_arrow": 25,
    "cast_fireball": 26,
    "cast_iceball": 27,
    "place_torch": 28,
    "drink_potion_red": 29,
    "drink_potion_green": 30,
    "drink_potion_blue": 31,
    "drink_potion_pink": 32,
    "drink_potion_cyan": 33,
    "drink_potion_yellow": 34,
    "read_book": 35,
    "enchant_sword": 36,
    "enchant_armour": 37,
    "make_torch": 38,
    "level_up_dexterity": 39,
    "level_up_strength": 40,
    "level_up_intelligence": 41,
    "enchant_bow": 42,
}
craftax_game_tips = [
    "Explore dungeons, mine resources, craft items, and fight enemies to progress through the game.",
    "Manage your health, hunger, thirst, energy, and mana to survive. Replenish them by eating, drinking, sleeping, and resting.",
    "Find and use ladders to descend to the next level after defeating 8 creatures on each floor.",
    "Start in the Overworld by gathering wood, crafting tools, and preparing for dungeon exploration.",
    "Craft tables and furnaces to create more advanced tools and weapons.",
    "Use the bow and arrows for ranged attacks, and craft arrows using wood and stone.",
    "Assign experience points to upgrade Dexterity, Strength, or Intelligence attributes.",
    "Experiment with different colored potions to discover their effects, which change each game.",
    "Use torches in dark levels to navigate and explore safely.",
    "Craft and wear armor to reduce damage from enemies.",
    "Learn and use spells for ranged magical attacks, especially against enemies resistant to physical damage.",
    "Enchant weapons and armor using gemstones and mana for elemental bonuses.",
    "Mine valuable ores like diamonds, sapphires, and rubies for crafting and enchanting.",
    "Adapt your strategy for each unique floor, considering the environment and enemy types.",
    "Build bridges across lava using stone in the Fire Realm.",
    "Prepare for the absence of food or water on certain levels by stocking up beforehand.",
    "Defeat the Necromancer in the final Graveyard level by surviving waves of enemies and attacking during vulnerable states.",
]

crafter_game_tips = [
    "Don't take actions for which you don't have the necessary requirements. For example, don't try to place a table if you don't have wood in your inventory.",
    "Place safety and survival first, but when the coast is clear strive to make progress on unlocking the next goals in your progression.",
]
