#library(mltools)
#library(data.table)

# function to change data format
cleanTitanicData_temp <- function(dat, dataType){
  
  # factorize Pclass variable
  dat$Pclass <- as.factor(dat$Pclass)  
  
  # extract titles from names
  dat$title <- str_split_fixed(as.character(str_split_fixed(dat$Name, ",", 2)[,2]),". ",2)[,1]
  
  # remove passenger name column
  dat <- dat[ , -which(names(dat) %in% c("Name"))]
  
  # NAs on Age replaced with median
  #dat$Age[is.na(dat$Age)] <- median(dat$Age[complete.cases(dat$Age)])
  
  # factorize cabin variable according to letter
  dat$Cabin <-sapply(dat$Cabin, function(x) substr(x,1,1))
  dat$Cabin[dat$Cabin==""] <- NA
  #dat$Cabin[grepl("F|G|T",dat$Cabin) == TRUE] <- "F"
  dat$Cabin <- as.factor(dat$Cabin)
  
  # factorize embarked variable according to letter
  #dat <- dat[dat$Embarked != "",]
  #dat <- dat[complete.cases(dat$Embarked),]
  dat$Embarked[dat$Embarked==""] <- NA
  dat$Embarked <- as.factor(dat$Embarked)
  
  # reduce factor size of ticket variable
  dat$Ticket <- as.character(dat$Ticket)
  dat$Ticket[grepl("L",dat$Ticket) == TRUE] <- "L"
  dat$Ticket[grepl("F",dat$Ticket) == TRUE] <- "F"
  dat$Ticket[grepl("PC|P",dat$Ticket) == TRUE] <- "P"
  dat$Ticket[grepl("CA|C.A.|CA.|C ",dat$Ticket) == TRUE] <- "C"
  dat$Ticket[grepl("A/5|A/5.|A.5.|A./5.|A/4.|A4.|A|A.|AQ",dat$Ticket) == TRUE] <- "A"
  dat$Ticket[grepl("S",dat$Ticket) == TRUE] <- "S"
  dat$Ticket[grepl("W",dat$Ticket) == TRUE] <- "W"
  #dat$Ticket <- sapply(dat$Ticket, function(x) ifelse(grepl("A|C|F|L|P|S|W",substring(x,1,1))==FALSE,substring(x,1,1),x))
  #dat$Ticket[grepl("1|2|3|A|C|P|S",dat$Ticket) == FALSE] <- "Other"
  dat$Ticket <- sapply(dat$Ticket, function(x) ifelse(nchar(x)>1,as.character(nchar(x)),x))
  #dat$Ticket[grepl("4|5|6|P|C|S",dat$Ticket) == FALSE] <- "Other"
  dat$Ticket <- as.factor(dat$Ticket)
  
  # Create family size variable including the passenger itself
  dat <- transform(dat, 'familySize' =  SibSp + Parch +1)
  dat <- dat[ , -which(names(dat) %in% c("SibSp","Parch"))]
  dat$Fsize <- NA
  dat$Fsize[dat$familySize == 1] <- 'singleton'
  dat$Fsize[dat$familySize <= 4 & dat$familySize > 1] <- 'small'
  dat$Fsize[dat$familySize > 4] <- 'large'
  dat <- dat[ , -which(names(dat) %in% c("familySize"))]
  dat$Fsize <- as.factor(dat$Fsize)
  
  # Log transform age and fare variable
  dat$AgeLog <- log(dat$Age)
  #dat$Fare[is.na(dat$Fare)] <- median(dat$Fare[complete.cases(dat$Fare)])
  dat$FareLog <- log(dat$Fare+1) # 1 added to avoid log(0)
  dat <- dat[ , -which(names(dat) %in% c("Age","Fare"))]
  
  # Create family size variable including the passenger itself
  dat$isMale <- NA
  dat$isMale[dat$Sex == "male"] <- 1
  dat$isMale[dat$Sex <= "female"] <- 0
  dat$isMale <- as.factor(dat$isMale)
  dat <- dat[ , -which(names(dat) %in% c("Sex"))]
  
  if (dataType == 'train') {
    # factorize Survived column
    dat$Survived <- as.factor(dat$Survived)
    dat$title[760] <- "Countess"
    dat$title[grepl("Mr|Mrs|Miss|Master",dat$title) == FALSE] <- "Other"
    dat$title <- as.factor(dat$title)
  } else{
    dat$title[grepl("Mr|Mrs|Miss|Master",dat$title) == FALSE] <- "Other"
    dat$title <- as.factor(dat$title)  
  }
  
  #Extract categorical variables
  #cat_var_names <- colnames(dat)[sapply(dat[,colnames(dat)],class) %in% c("factor","character")]
  #dat_cat <- dat[,cat_var_names]
  
  #if (dataType == 'train') {
  ## one hot encoding except for survived and isMale columns
  #dat <- one_hot(as.data.table(dat), cols = cat_var_names[cat_var_names != c("Survived","isMale")])
  #} else{
  #  dat <- one_hot(as.data.table(dat), cols = cat_var_names[cat_var_names != c("isMale")])  
  #}
  
  return(dat)
}