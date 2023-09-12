import random




def main():
    rand_num  = random.randint(1,100)
    score = 100
    while True:
        user_guess = input('guess a number between 1 and 100 : ')
        if user_guess == 'q':
            print('Thank you for playing. Goodbye. ')
            break
        elif not user_guess.isdigit():
            print('Invalid Input. Please try again.')
            continue
    
        user_guess = int(user_guess)
        if user_guess > 100 or user_guess < 1:
            print('Invalid Input. Your guess should be between 1 and 100.')
            continue

        if rand_num > user_guess:
            print('your guess is too low. please try again.')
        elif rand_num < user_guess:
            print('your guess is too high. please try again.')
        else:
            print('congrats! you guessed the correct number')
            print(f'Your score is : {score} ')
            break

        score -= 10
        score = max(score , 0)



     

    



if __name__ == '__main__':
    main()


