use spearmint
db['myconstraint.jobs'].remove({status:'pending'})
db['myconstraint.jobs'].remove({status:'complete'})
db.dropDatabase()
exit